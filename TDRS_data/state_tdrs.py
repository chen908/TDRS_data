import torch
from typing import NamedTuple
# windows
from boolmask import mask_long2bool

# linux
# from boolmask import mask_long2bool
class StateTDRS(NamedTuple):
    # Fixed input
    task: torch.Tensor  # window1, window2, window3
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the task_info and ava_win tensors are not kept multiple times
    # so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    rs: torch.Tensor  # number of relay_satellites

    # State
    prev_a: torch.Tensor
    cur_task: torch.Tensor
    cur_t: torch.Tensor
    no_window: torch.Tensor
    priority: torch.Tensor
    total_priority: torch.Tensor
    unfilled_priority: torch.Tensor
    negative_priority: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    i: torch.Tensor  # Keeps track of step
    # __getitemraw__ = tuple.__getitem__


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                rs=self.rs[key],
                prev_a=self.prev_a[key],
                cur_task=self.cur_task[key],
                cur_t=self.cur_t[key],
                no_window=self.no_window[key],
                priority=self.priority[key],
                total_priority=self.total_priority[key],
                unfilled_priority=self.unfilled_priority[key],
                negative_priority=self.negative_priority[key],
                visited_=self.visited_[key],


            )
        # return super(StateTDRS, self).__getitem__(key)
        # return self.__getitemraw__(key)
        return tuple.__getitem__(self, key)


    @staticmethod
    def initialize(input):

        task = input['task']
        virtual = input['virtual']
        batch_size, n_task, info = task.size()
        rs_number = int((info-2)/2)
        total_priority = task[:, :, 7].sum(1)[:, None]
        priority_1 = task[:, :, 7][:, None]
        priority_2 = task[:, :, 7][:, None]
        priority_3 = task[:, :, 7][:, None]
        priority = torch.cat((priority_1, priority_2, priority_3), 1)
        un_1 = task[torch.arange(batch_size)[:, None], :, 2*torch.zeros(batch_size, dtype=torch.long, device=task.device)[:, None]] == torch.zeros(batch_size, 1, n_task, dtype=torch.long, device=task.device)
        un_2 = task[torch.arange(batch_size)[:, None], :, 2*torch.ones(batch_size, dtype=torch.long, device=task.device)[:, None]] == torch.zeros(batch_size, 1, n_task, dtype=torch.long, device=task.device)
        un_3 = task[torch.arange(batch_size)[:, None], :, 4*torch.ones(batch_size, dtype=torch.long, device=task.device)[:, None]] == torch.zeros(batch_size, 1, n_task, dtype=torch.long, device=task.device)
        no_window = torch.cat((un_1, un_2, un_3), 1)
        priority[no_window] = 0
        unfilled_priority = priority.sum(-1)/total_priority
        return StateTDRS(
            task=torch.cat((virtual, task), -2),
            ids=torch.arange(batch_size, dtype=torch.int64, device=task.device)[:, None],
            rs=torch.arange(rs_number, dtype=torch.int64, device=task.device),
            prev_a=torch.zeros(batch_size, rs_number, dtype=torch.long, device=task.device),
            cur_task=torch.zeros(batch_size, rs_number, info, device=task.device),
            cur_t=torch.zeros(batch_size, rs_number, device=task.device),
            no_window=no_window,
            priority=priority,
            total_priority=total_priority,
            unfilled_priority=unfilled_priority,
            negative_priority=torch.zeros(batch_size, rs_number, device=task.device),
            visited_=torch.zeros(batch_size, rs_number, n_task + 1, dtype=torch.uint8, device=task.device),
            i=torch.zeros(1, dtype=torch.int64, device=task.device)
        )

    def get_final_reward(self):

        return self.negative_priority

    def update(self, selected, rs):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        negative_priority = self.negative_priority
        cur_t = self.cur_t
        visited_ = self.visited_
        priority = self.priority
        prev_a = selected
        batch_size, rs_num = selected.size()
        cur_task = self.task[self.ids, selected]
        for i in range(batch_size):
            negative_priority[i, rs[i]] += -cur_task[i, rs[i], 7]
            if cur_task[i, rs[i], 7] > 0:
                cur_t[i, rs[i]] += (cur_task[i, rs[i], 6] + 1/144)
            else:
                cur_t[i, rs[i]] += cur_task[i, rs[i], 6]
            visited_[i, rs[i], prev_a[i, rs[i]]] = 1
            visited_[i, rs[i], 0] = 0  # virtual node can be selected all the time
        # exceeds_time = self.task[self.ids, 1:, 2*rs[:, None]+1] <= self.cur_t[self.ids, rs[:, None]][..., None].expand_as(
        #     self.task[self.ids, 1:, 2*rs[:, None]])
        priority[visited_.sum(1).unsqueeze(1)[:, :, 1:].to(self.no_window.dtype) | self.no_window] = 0
        unfilled_priority = priority.sum(-1)/self.total_priority
        return self._replace(
            prev_a=prev_a, cur_task=cur_task, cur_t=cur_t, unfilled_priority=unfilled_priority, negative_priority=negative_priority
            , visited_=visited_, i=self.i + 1
        )

    def get_current_node(self):
        return self.prev_a

    def get_current_time(self):
        return self.cur_t

    def get_current_unfilled(self):
        return self.unfilled_priority

    def get_mask(self, rs):
        """
        Gets a (batch_size, n_window) mask with the feasible actions, depends on already visited and
        corresponding task number. 0 = feasible, 1 = infeasible
        :return:
        """
        batch_size = self.visited_.size(0)
        visited = self.visited_.sum(1).unsqueeze(1)[:, :, 1:]
        # a = self.task[self.ids, 1:, 2*rs[:, None]]
        unreached_time = self.task[self.ids, 1:, 2*rs[:, None]] >= self.cur_t[self.ids, rs[:, None]][..., None].expand_as(
            self.task[self.ids, 1:, 2*rs[:, None]]
        )
        # a = self.task[self.ids, 1:, 2*rs[:, None]]
        # b = self.task[self.ids, 1:, 6]
        exceeds_time = self.task[self.ids, 1:, 2*rs[:, None]+1] <= self.cur_t[self.ids, rs[:, None]][..., None].expand_as(
            self.task[self.ids, 1:, 2*rs[:, None]]
        )
        mask_task = visited.to(unreached_time.dtype) | unreached_time | exceeds_time
        # for i in range(mask_task.size(0)):
        #     for j in range(mask_task.size(1)):
        #         if (self.cur_t[i, rs[i]] < self.task[i, j, 2*rs[i]]) or (self.cur_t[i, rs[i]] > self.task[i, j, 2*rs[i]+1]):
        #             mask_task[i, j] = True
        #     mask_task[i, 0] = False
        mask_virtual = torch.zeros(batch_size, 1, dtype=bool, device=self.visited_.device)
        return torch.cat((mask_virtual[:, :, None], mask_task), -1)

    def all_finished(self):
        return (self.cur_t >= 0.5).all()


