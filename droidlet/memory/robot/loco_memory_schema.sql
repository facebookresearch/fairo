-- Copyright (c) Facebook, Inc. and its affiliates.


PRAGMA foreign_keys = ON;

CREATE TABLE DetectedObjectFeatures(
    uuid            NCHAR(36)   NOT NULL,
    featureBlob     BLOB,
    minx        FLOAT,
    miny        FLOAT,
    minz        FLOAT,
    maxx        FLOAT,
    maxy        FLOAT,
    maxz        FLOAT,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TRIGGER DetectedObjUpdate AFTER UPDATE ON DetectedObjectFeatures
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;

CREATE TABLE HumanPoseFeatures(
    uuid            NCHAR(36)   NOT NULL,
    keypointsBlob     BLOB,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TRIGGER HumanPoseUpdate AFTER UPDATE ON HumanPoseFeatures
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;
