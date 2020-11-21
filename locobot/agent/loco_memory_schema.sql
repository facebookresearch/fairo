-- Copyright (c) Facebook, Inc. and its affiliates.


PRAGMA foreign_keys = ON;

CREATE TABLE DetectedObjectFeatures(
    uuid            NCHAR(36)   NOT NULL,
    featureBlob     BLOB,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TABLE HumanPoseFeatures(
    uuid            NCHAR(36)   NOT NULL,
    keypointsBlob     BLOB,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);