"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
if __name__ == "__main__":
    import os
    import sys
    import torch
    import argparse
    from shape_transform_dataloader import ModifyData
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=1000000, help="num examples to build")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--max_meta", type=int, default=20, help="max meta")
    parser.add_argument("--sidelength", type=int, default=32, help="sidelength for dataloader")
    parser.add_argument(
        "--load_dictionary",
        default="/private/home/aszlam/junk/word_modify_word_ids.pk",
        help="where to get word dict",
    )
    parser.add_argument("--ndonkeys", type=int, default=8, help="workers in dataloader")
    args = parser.parse_args()
    this_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join(this_dir, "../")
    sys.path.append(parent_dir)

    ################# FIXME!!!!
    args.allow_same = False
    args.debug = -1
    args.words_length = 12
    args.sidelength = args.sidelength
    args.nexamples = args.num_examples + 10000
    args.tform_weights = {
        "thicker": 1.0,
        "scale": 1.0,
        "rotate": 1.0,
        "replace_by_block": 1.0,
        #                          'replace_by_n': 1.0,
        "replace_by_halfspace": 1.0,
        "fill": 1.0,
    }

    train_data = ModifyData(args, dictionary=args.load_dictionary)

    num_workers = args.ndonkeys

    print("making dataloader")
    rDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=args.ndonkeys,
    )

    X = torch.zeros(
        args.num_examples, args.sidelength, args.sidelength, args.sidelength, dtype=torch.int
    )
    Y = torch.zeros(
        args.num_examples, args.sidelength, args.sidelength, args.sidelength, dtype=torch.int
    )
    words = torch.zeros(args.num_examples, args.words_length)
    it = iter(rDL)
    for i in tqdm(range(args.num_examples // 32 + 1)):
        b = it.next()
        c = min((i + 1) * 32, args.num_examples)
        X[i * 32 : c] = b[1]
        Y[i * 32 : c] = b[2]
        words[i * 32 : c] = b[0]
