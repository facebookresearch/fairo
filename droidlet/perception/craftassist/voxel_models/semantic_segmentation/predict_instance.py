import numpy as np

from vision import SemSegWrapper


SL = 17 
H = 13

model_path = "/checkpoint/yuxuans/jobs/hitl_vision/data_shapes_6kind_6000_nfbid_0_nepochs_500_lr_0.001_batchsz_256_sampleEmptyProb_0.05_hiddenDim_128_noTargetProb_0.3_probThreshold_0.3_queryEmbed_bert_runName_SWEEP2/model.pt"

if __name__ == "__main__":
    fake_data = np.zeros((SL, H, SL), dtype="int32")
    # a cube
    (x, y, z) = (0, 0, 0)
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                fake_data[x + ix, y + iy, z + iz] = 50

    # a bar
    (x, y, z) = (7, 0, 7)
    for ix in range(1):
        for iy in range(10):
            for iz in range(1):
                fake_data[x + ix, y + iy, z + iz] = 51

    # a sphere
    # N = 6
    # c = N / 2 - 1 / 2
    # CNT = 0
    # for r in range(N):
    #     for s in range(N):
    #         for t in range(N):
    #             w = ((r - c) ** 2 + (s - c) ** 2 + (t - c) ** 2) ** 0.5
    #             if w < N / 2:
    #                 fake_data[r, s, t] = 50
    #                 # S.append(((r, s, t), bid))
    #                 CNT += 1
    # print(f"CNT: {CNT}")
    
    for ix in range(17):
        for iy in range(10, 12):
            for iz in range(17):
                fake_data[ix, iy, iz] = 46

    text = "where is the cube"

    model = SemSegWrapper(model=model_path)
    pred = model.perceive(fake_data, text_span=text, offset=None)

    print(pred)



    
