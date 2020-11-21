-- Copyright (c) Facebook, Inc. and its affiliates.


PRAGMA foreign_keys = ON;


CREATE TABLE Memories (
    uuid                    NCHAR(36)       PRIMARY KEY,
    node_type               VARCHAR(255)    NOT NULL,
    create_time             INTEGER         NOT NULL,
    updated_time            INTEGER         NOT NULL,
    attended_time           INTEGER         NOT NULL DEFAULT 0,
    is_snapshot             BOOLEAN         NOT NULL DEFAULT FALSE
);


CREATE TABLE Chats (
    uuid    NCHAR(36)       PRIMARY KEY,
    speaker VARCHAR(255)    NOT NULL,
    chat    TEXT            NOT NULL,
    time    INTEGER         NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ChatsTime ON Chats(time);

CREATE TABLE ReferenceObjects (
    uuid        NCHAR(36)       PRIMARY KEY,
    eid         INTEGER,
    x           FLOAT,
    y           FLOAT,
    z           FLOAT,
    yaw         FLOAT,         -- look vec
    pitch       FLOAT,         -- look vec
    name        VARCHAR(255),  -- for people/player/agent names, etc.
    type_name   VARCHAR(255),  -- for mob types in mc, etc.
    ref_type    VARCHAR(255),  -- what kind of ref object is this?

    UNIQUE(eid),
    UNIQUE(uuid),
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX RefObjXYZ ON ReferenceObjects(x, y, z);

CREATE TABLE ArchivedReferenceObjects (
    uuid        NCHAR(36)       PRIMARY KEY,
    eid         INTEGER,
    x           FLOAT,
    y           FLOAT,
    z           FLOAT,
    yaw         FLOAT,         -- look vec
    pitch       FLOAT,         -- look vec
    name        VARCHAR(255),  -- for people/player/agent names, etc.
    type_name   VARCHAR(255),  -- for mob types in mc, etc.
    ref_type    VARCHAR(255),  -- what kind of ref object is this?

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ArchivedRefObjXYZ ON ArchivedReferenceObjects(x, y, z);


CREATE TABLE Times (
    uuid    NCHAR(36)       PRIMARY KEY,
    time    INTEGER           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);


CREATE TABLE Programs (
    uuid          NCHAR(36)       PRIMARY KEY,
    logical_form  TEXT            NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);


CREATE TABLE Tasks (
    uuid        NCHAR(36)       PRIMARY KEY,
    action_name VARCHAR(32)     NOT NULL,
    pickled     BLOB            NOT NULL,
    paused      BOOLEAN         NOT NULL DEFAULT 0,
    created_at  INTEGER         NOT NULL,
    finished_at INTEGER         NOT NULL DEFAULT -1,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX TasksFinishedAt ON Tasks(finished_at);


CREATE TABLE Dances (
    uuid      NCHAR(36)       PRIMARY KEY
);

-- TODO! make sure subj_text matches the name field of the subj NamedAbstraction
-- if subj is a NamedAbstraction, and make sure subj is a NamedAbstraction
-- if subj_text is given
-- same for obj_text
CREATE TABLE Triples (
    uuid             NCHAR(36)       PRIMARY KEY,
    subj             NCHAR(36)       NOT NULL,  -- memid of subj, could be ReferenceObject, Chat, NamedAbstraction or other
    subj_text        TEXT,
    pred             NCHAR(36),			-- memid of NamedAbstraction
    pred_text        TEXT,                      -- has_tag_, has_name_, etc.  should be the name of a NamedAbstraction
    obj              NCHAR(36),                 -- memid of obj, could be ReferenceObject, Chat, NamedAbstraction or other
    obj_text         TEXT,
    confidence       FLOAT           NOT NULL DEFAULT 1.0,

    UNIQUE(subj, pred, obj) ON CONFLICT REPLACE,
    FOREIGN KEY(subj) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX TriplesSubjPred ON Triples(subj, pred);
CREATE INDEX TriplesPredObj ON Triples(pred, obj);
CREATE INDEX TriplesSubjPredText ON Triples(subj, pred_text);
CREATE INDEX TriplesPredTextObjText ON Triples(pred_text, obj_text);


CREATE TABLE NamedAbstractions(
    uuid    NCHAR(36)       PRIMARY KEY,
    name    TEXT            NOT NULL,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);


CREATE TABLE SetMems(
    uuid    NCHAR(36)       PRIMARY KEY,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
