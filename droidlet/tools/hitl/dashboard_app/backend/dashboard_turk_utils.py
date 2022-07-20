from droidlet.tools.crowdsourcing.sync_whitelists import (
    add_workers_to_quals,
    compare_qual_lists,
    import_s3_lists,
    pull_local_lists,
    revoke_worker_qual,
    update_lists,
    qual_dict,
)

S3_BUCKET_NAME_INTERNAL = "droidlet-internal"

PIPELINE_QUAL_MAPPPING = {
    "NLU": ["interaction"],
}

QUAL_LIST_ORDER = {"allow": 0, "block": 1, "softblock": 2}


def get_turk_list_by_pipeline(pipeline):
    """
    Download turk allow/block/softblock list from local mephisto db,
    return lists corresponding the the input pipeline
    """
    output_dict_raw = pull_local_lists()
    qual_types = PIPELINE_QUAL_MAPPPING[pipeline]

    output_dict = {}

    for qual_type in qual_types:
        output_dict[qual_type] = output_dict_raw[qual_type]
    return output_dict


def update_turk_qual_by_tid(tid, task_type, new_list_type, prev_list_type):
    """
    Update the local mephisto db
    """
    if QUAL_LIST_ORDER[new_list_type] > QUAL_LIST_ORDER[prev_list_type]:
        success = add_workers_to_quals([tid], qual_dict[task_type][new_list_type])
    elif QUAL_LIST_ORDER[new_list_type] < QUAL_LIST_ORDER[prev_list_type]:
        for qual in QUAL_LIST_ORDER:
            # remove all qual between (new, prev]
            success = True
            if (
                QUAL_LIST_ORDER[qual] > QUAL_LIST_ORDER[new_list_type]
                and QUAL_LIST_ORDER[qual] <= QUAL_LIST_ORDER[prev_list_type]
            ):
                success &= revoke_worker_qual(tid, qual_dict[task_type][qual])
    else:
        # no change
        success = True

    if success:
        print(pull_local_lists())
        return "ok", None
    else:
        return (
            f"Cannot update the worker: {tid} to {new_list_type} from {prev_list_type} of qual: {task_type} ",
            400,
        )


def update_turk_list_to_sync():
    """
    sync local turk list with s3
    """
    s3_list_dict = import_s3_lists(S3_BUCKET_NAME_INTERNAL)
    local_list_dict = pull_local_lists()

    diff_dict = compare_qual_lists((s3_list_dict, local_list_dict))

    update_lists(S3_BUCKET_NAME_INTERNAL, diff_dict)
