class LoadData:
    def __init__(self, user_set,
                 item_set,
                 train_pt,
                 train_list,
                 test_cold_gt,
                 test_cold_samples,
                 test_cold_u,
                 test_samples,
                 test_warm_gt,
                 test_warm_samples,
                 test_warm_u,
                 val_cold_gt,
                 val_cold_samples,
                 val_cold_u,
                 val_warm_gt,
                 val_warm_samples,
                 val_warm_u,
                 val_samples,
                 val_gt,
                 test_gt,
                 hard_negatives):
        super(LoadData, self).__init__()
        self.user_set = user_set
        self.item_set = item_set
        self.train_pt = train_pt
        self.train_list = train_list
        self.test_cold_gt = test_cold_gt
        self.test_cold_samples = test_cold_samples
        self.test_cold_u = test_cold_u
        self.test_samples = test_samples
        self.test_warm_gt = test_warm_gt
        self.test_warm_samples = test_warm_samples
        self.test_warm_u = test_warm_u
        self.val_cold_gt = val_cold_gt
        self.val_cold_samples = val_cold_samples
        self.val_cold_u = val_cold_u
        self.val_warm_gt = val_warm_gt
        self.val_warm_samples = val_warm_samples
        self.val_warm_u = val_warm_u
        self.val_samples = val_samples
        self.val_gt = val_gt
        self.test_gt = test_gt
