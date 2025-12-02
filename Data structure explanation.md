            +-------------------------------------------+
            |      base_dataset (transform=None)        |
            |  - Responsible for reading raw image files
            return image + label   |
            +-------------------------------------------+
                                 |
                                 |   random_split(train, val)
                                 v
        +---------------------+         +----------------------+
        |     train_subset    |         |      val_subset      |
        | (image + index)   |           |  (image + index) ee   |
        +---------------------+         +----------------------+
                   |                                |
                   | use diff train_transform       |use fixed val_transform
                   v                                v
   +--------------------------------+   +----------------------------------+
   |      train_dataset_xxx         |   |          val_dataset            |
   |  (Training Dataset with 
        Random Augmentation)     |   | (only resize + normalize，no aug) |
   +--------------------------------+   +----------------------------------+
                   |                                |
                   | build dif train DataLoader     |build val DataLoader
                   v                                v
       +-------------------------+        +-------------------------+
       |    train_loader_xxx     |        |        val_loader       |
       | - batch                |          - no random aug          |
       | - shuffle / sampler     |        | - for evaluation       |
       +-------------------------+        +-------------------------+







