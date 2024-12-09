def mask_ratio_schedule(cur_epoch: int, max_epoch: int, max_mask_ratio: float, mask_ratio_scheduler_type: str) -> float:
    if mask_ratio_scheduler_type == 'interval':
        if cur_epoch <= 1/20 * max_epoch:
            return 1/4 * max_mask_ratio
        elif cur_epoch <= 1/5 * max_epoch:
            return 2/4 * max_mask_ratio
        elif cur_epoch <= 1/2 * max_epoch:
            return 3/4 * max_mask_ratio
        else:
            return max_mask_ratio
    elif mask_ratio_scheduler_type == 'linear':
        raise NotImplementedError()
    elif mask_ratio_scheduler_type == None:
        return max_mask_ratio
    else:
        raise ValueError(f"Unknown mask_ratio_scheduler_type: {mask_ratio_scheduler_type}")