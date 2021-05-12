"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
if __name__ == "__main__":
    import os
    import sys
    import torch
    import argparse
    import conv_models as models
    from shape_transform_dataloader import ModifyData
    from voxel_models.plot_voxels import SchematicPlotter, draw_rgb  # noqa
    import visdom
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=1000000, help="num examples to encode")
    parser.add_argument("--color_io", action="store_true", help="input uses color-alpha")
    parser.add_argument(
        "--max_meta", type=int, default=20, help="allow that many meta values when hashing idmeta"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--model_filepath", default="", help="from where to load model")
    parser.add_argument(
        "--load_dictionary",
        default="/private/home/aszlam/junk/word_modify_word_ids.pk",
        help="where to get word dict",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="step size for net")
    parser.add_argument(
        "--sbatch", action="store_true", help="cluster run mode, no visdom, formatted stdout"
    )
    parser.add_argument("--ndonkeys", type=int, default=8, help="workers in dataloader")
    args = parser.parse_args()
    if not args.sbatch:
        vis = visdom.Visdom(server="http://localhost")
        sp = SchematicPlotter(vis)

    print("loading train data")

    ################# FIXME!!!!
    args.allow_same = False
    args.debug = -1
    args.words_length = 12
    args.sidelength = 32
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

    print("making models")
    args.num_words = train_data.padword + 1
    args.word_padding_idx = train_data.padword

    model = models.AE(args, args.model_filepath)
    model.cuda()
    model.eval()
    X = None
    it = iter(rDL)
    with torch.no_grad():
        for i in tqdm(range(args.num_examples // 32 + 1)):
            b = it.next()
            words = b[0]
            x = b[1]
            y = b[2]
            x = x.cuda()
            y = y.cuda()
            z = model(x)
            if X is None:
                szs = model.hidden_state.shape
                X = torch.zeros(args.num_examples, szs[1], szs[2], szs[3], szs[4])
                Y = torch.zeros(args.num_examples, szs[1], szs[2], szs[3], szs[4])
                all_words = torch.LongTensor(args.num_examples, 12)
            c = min((i + 1) * 32, args.num_examples)
            X[i * 32 : c] = model.hidden_state
            z = model(y)
            Y[i * 32 : c] = model.hidden_state
            all_words[i * 32 : c] = words
