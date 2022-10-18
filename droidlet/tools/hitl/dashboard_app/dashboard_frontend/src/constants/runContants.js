/*
Copyright (c) Facebook, Inc. and its affiliates.

Constants used for the run detial components.
*/
export const METADATA_CONSTANTS = {
    "BATCH_ID": { label: "Batch ID", span: 1},
    "NAME": { label: "Name", span: 1},
    "S3_LINK": { label: "AWS S3 Link", span: 3},
    "COST": { label: "Cost", span: 1},
    "START_TIME": { label: "Start Time", span: 1},
    "END_TIME": { label: "End Time", span: 1},
}

export const METADATA_ORDER = [
    "BATCH_ID", "NAME", "COST", "S3_LINK",
    "START_TIME", "END_TIME",
]

export const JOB_TYPES = {
    "INTERACTION": "Interaction",
    "ANNOTATION": "Annotation",
    "RETRAIN": "Retrain"
}

export const JOB_STATUS_CONSTANTS = {
    "STATUS": { label: "Status", span: 2},
    "START_TIME": { label: "Start Time", span: 2},
    "END_TIME": { label: "End Time", span: 2},
    "ENABLED": { label: "Enabled", span: 2},
    "NUM_COMPLETED": { label: "Number of Complted Jobs", span: 2},
    "NUM_REQUESTED": { label: "Number of Requested Jobs", span: 2},
    "NUM_ERR_COMMAND": { label: "Number of Error Commands", span: 2},
    "NUM_SESSION_LOG": { label: "Number of Session Logs", span: 4},
    "NUM_COMMAND": { label: "Number of Total Commands", span: 2},
    "DASHBOARD_VER": { label: "Dashboard Version (sha256)", span: 4},
    "MODEL_LOSS": { label: "Loss", span: 1},
    "NEW_DATA_SZ": { label: "New Data Size", span: 1},
    "MODEL_ACCURACY": { label: "Accuracy", span: 1},
    "ORI_DATA_SZ": { label: "Original Data Size", span: 2},
    "MODEL_EPOCH": { label: "Epoch", span: 1}
}

export const JOB_STATUS_ORDER = [
    "STATUS",
    "ENABLED",
    "DASHBOARD_VER",
    "START_TIME", "END_TIME",
    "NUM_REQUESTED", "NUM_COMPLETED",
    "NUM_COMMAND", "NUM_ERR_COMMAND",
    "NUM_SESSION_LOG",
    "MODEL_LOSS", "MODEL_ACCURACY", "MODEL_EPOCH",
    "ORI_DATA_SZ", "NEW_DATA_SZ",
]