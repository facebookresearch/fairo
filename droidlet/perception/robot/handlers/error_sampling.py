"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import json
import random
import time


class ErrorSampler:
    def __init__(self, task_name, sample_rate=1):
        self.task_name = task_name
        self.path = "SampledErrors/" + task_name
        self.sample_rate = sample_rate

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __call__(self, raw_image, prob_list, metadata):
        if self._random_sample():
            timeStamp = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
            imgFilename = self.path + f"/BotCapture_{timeStamp}.jpg"
            if raw_image:
                raw_image.save(imgFilename)  # TODO: this may be different for different models
            with open(self.path + "/ModelData.json", "a+") as jsonFile:
                jsonObj = {"filename": imgFilename, "data": metadata}
                metajson = json.dumps(jsonObj)
                jsonFile.write(metajson)

    def _random_sample(self):
        return True if random.random() < self.sample_rate else False

    def _cusp_sampling(self, prob_list, cutoff=0.8, spread=0.1, sample_rate=0.05):
        count = sum(
            map(lambda x: 1 if x > cutoff - spread and x < cutoff + spread else 0, prob_list)
        )
        willSample = False
        for i in range(count):
            if self._random_sample(sample_rate):
                willSample = True
        return willSample
