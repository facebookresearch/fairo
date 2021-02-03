-- Copyright (c) Facebook, Inc. and its affiliates.

-- todo player_placed is string with player eid

ALTER TABLE ReferenceObjects
ADD player_placed BOOLEAN;
ALTER TABLE ReferenceObjects
ADD agent_placed BOOLEAN;
ALTER TABLE ReferenceObjects
ADD created INTEGER;
ALTER TABLE ReferenceObjects
ADD updated INTEGER;
ALTER TABLE ReferenceObjects
ADD voxel_count INTEGER;



ALTER TABLE ArchivedReferenceObjects
ADD player_placed BOOLEAN;
ALTER TABLE ArchivedReferenceObjects
ADD agent_placed BOOLEAN;
ALTER TABLE ArchivedReferenceObjects
ADD created INTEGER;
ALTER TABLE ArchivedReferenceObjects
ADD updated INTEGER;
ALTER TABLE ArchivedReferenceObjects
ADD voxel_count INTEGER;



CREATE TABLE BlockTypes (
    uuid            NCHAR(36)   NOT NULL,
    type_name       VARCHAR(32) NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);



CREATE TABLE MobTypes (
    uuid            NCHAR(36)   NOT NULL,
    type_name       VARCHAR(32) NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);



-- TODO player_placed is string with player id
CREATE TABLE VoxelObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT,
    meta            TINYINT,
    agent_placed    BOOLEAN,
    player_placed   BOOLEAN,
    updated         INTEGER,
    ref_type        varchar(255),

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX VoxelObjectsXYZ ON VoxelObjects(x, y, z);
CREATE TRIGGER VoxelObjectsDelete AFTER DELETE ON VoxelObjects
    WHEN ((SELECT COUNT(*) FROM VoxelObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER VoxelObjectsUpdateCheckDeleted AFTER UPDATE ON VoxelObjects
    WHEN ((SELECT COUNT(*) FROM VoxelObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER VoxelObjectsUpdate AFTER UPDATE ON VoxelObjects
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;
--if a block is deleted, mark the uuid as updated, not deleted
--if all the blocks are deleted, VoxelObjectsDelete and then MemoryRemoved Triggers will fire
CREATE TRIGGER VoxelObjectsBlockDelete AFTER DELETE ON VoxelObjects
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;


CREATE TABLE ArchivedVoxelObjects (
    uuid            NCHAR(36)      NOT NULL,
    x               INTEGER        NOT NULL,
    y               INTEGER        NOT NULL,
    z               INTEGER        NOT NULL,
    bid             TINYINT,
    meta            TINYINT,
    agent_placed    BOOLEAN,
    player_placed   BOOLEAN,
    updated         INTEGER,
    ref_type        varchar(255),

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ArchivedVoxelObjectsXYZ ON ArchivedVoxelObjects(x, y, z);
CREATE TRIGGER ArchivedVoxelObjectsDelete AFTER DELETE ON ArchivedVoxelObjects
    WHEN ((SELECT COUNT(*) FROM ArchivedVoxelObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER ArchivedVoxelObjectsUpdate AFTER UPDATE ON ArchivedVoxelObjects
    WHEN ((SELECT COUNT(*) FROM ArchivedBlockObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed


CREATE TABLE Schematics (
    uuid    NCHAR(36)   NOT NULL,
    x       INTEGER     NOT NULL,
    y       INTEGER     NOT NULL,
    z       INTEGER     NOT NULL,
    bid     TINYINT     NOT NULL,
    meta    TINYINT     NOT NULL,

    UNIQUE(uuid, x, y, z),
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX SchematicsXYZ ON Schematics(x, y, z);


CREATE TABLE Mobs (
    uuid          NCHAR(36)       PRIMARY KEY,
    eid           INTEGER         NOT NULL,
    x             FLOAT           NOT NULL,
    y             FLOAT           NOT NULL,
    z             FLOAT           NOT NULL,
    mobtype       VARCHAR(255)    NOT NULL,
    player_placed BOOLEAN         NOT NULL,
    agent_placed  BOOLEAN         NOT NULL,
    spawn         INTEGER         NOT NULL,

--    UNIQUE(eid), not unique because of archives.  make a MobArchive table?
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TRIGGER MobsUpdate AFTER UPDATE ON Mobs
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;

CREATE TABLE Rewards (
    uuid    NCHAR(36)       PRIMARY KEY,
    value   VARCHAR(32)     NOT NULL, -- {POSITIVE, NEGATIVE}
    time    INTEGER         NOT NULL,
    
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TRIGGER RewardsUpdate AFTER UPDATE ON Rewards
    BEGIN INSERT INTO Updates(uuid, update_type) VALUES (OLD.uuid, 'update');
END;
