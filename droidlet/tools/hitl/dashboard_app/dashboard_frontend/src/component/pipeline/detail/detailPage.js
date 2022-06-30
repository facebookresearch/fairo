/*
Copyright (c) Facebook, Inc. and its affiliates.

Detail page of a run.

Batach ID and pipeline type needs to be specified by the caller. 

Usage:
<DetailPage batchId={batchId} pipelineType={pipelineType} />
*/
import { Button, Card, Collapse, Divider, Timeline, Typography } from "antd";
import React, { useState } from "react";
import {
    ClockCircleOutlined,
    CheckOutlined,
    LoadingOutlined,
    CaretRightOutlined
} from '@ant-design/icons';
import { Link, Outlet, useParams } from "react-router-dom";
import { toFirstCapital } from "../../../utils/textUtils";
import { TAB_ITEMS } from "../../../constants/pipelineConstants";
import MetaInfoDescription from "./metaInfoDescription";

const { Title } = Typography;
const { Panel } = Collapse;

const DetailPage = (props) => {
    const pipelineType = props.pipelineType;
    const batch_id = useParams().batch_id;

    const fakeRunInfo = JSON.parse(`{"BATCH_ID": 20220621142744, 
    "NAME": "cw_test_jm_1", 
    "S3_LINK": "https://s3.console.aws.amazon.com/s3/buckets/droidlet-hitl?region=us-west-1&prefix=20220621142744", "COST": null, "START_TIME": "2022-06-21 14:27:44.352281", "END_TIME": null, 
    "INTERACTION": {"STATUS": "DONE", "END_TIME": ["2022-06-22 14:28:44.552815"], "START_TIME": ["2022-06-21 14:27:44.552815"], "NUM_COMPLETED": null, "ENABLED": null, "NUM_REQUESTED": null, "NUM_ERR_COMMAND": null, "NUM_SESSION_LOG": null, "NUM_COMMAND": null, "DASHBOARD_VER": "sha256:19682b1dcdd336a988879fa457383341e34629e042e9799616d7188f10a561e5"}, 
    "ANNOTATION": {"STATUS": "RUNNING", "END_TIME": null, "START_TIME": null, "NUM_COMPLETED": null, "ENABLED": null, "NUM_REQUESTED": null}, 
    "RETRAIN": {"STATUS": null, "END_TIME": null, "START_TIME": null, "NUM_COMPLETED": null, "ENABLED": null, "NUM_REQUESTED": null, "ORI_DATA_SZ": null, "NEW_DATA_SZ": null, "MODEL_ACCURACY": null}}`)

    const getJobs = (runInfo) => (
        Object
            .entries(fakeRunInfo)
            .filter((obj) => typeof (obj[1]) === 'object' && obj[1]) // is a job info if is an object and is not null
    );

    const jobs = getJobs(fakeRunInfo);
    const [selectedJob, setSelectedJob] = useState(jobs[0][0]);

    const getIconForStatus = (jobStatus) => {
        if (!jobStatus) {
            // not started 
            return <ClockCircleOutlined />
        } else if (jobStatus === 'DONE') {
            return <CheckOutlined />
        } else {
            // running
            return <LoadingOutlined />
        }
    };

    const getJobButton = (job) => {
        const jobKey = job[0];
        const jobName = `${toFirstCapital(jobKey)} Jobs`

        if (job[1].STATUS) {
            // running or finished
            return <Button type="link" value={jobName}><Link to={jobKey.toLocaleLowerCase()}>{jobName}</Link></Button>;
        } else {
            return <p>{jobName}</p>;
        }
    }

    return <div style={{ 'padding': '0 24px 0 24px' }}>
        <Collapse
            bordered={false}
            expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
            defaultActiveKey={['overview']}
        >
            <Panel
                header={<Title level={4}>{`Overview of Run ${batch_id}`}</Title>}
                key='overview'
            >
                <MetaInfoDescription metaInfo={fakeRunInfo}/>
            </Panel>
        </Collapse>

        <Divider />
        <div style={{ 'display': 'flex' }}>
            <div style={{ 'width': '160px' }}>
                <Timeline>
                    {
                        jobs.map((job) =>
                            <Timeline.Item dot={getIconForStatus(job[1].STATUS)}>
                                {getJobButton(job)}
                            </Timeline.Item>
                        )
                    }
                </Timeline>
            </div>
            <Outlet context={{ metaInfo: fakeRunInfo }} />
        </div>
        <Button type="primary">
            <Link to="../" state={{ label: TAB_ITEMS.RUNS.label, key: TAB_ITEMS.RUNS.key }}>
                Back to View All
            </Link>
        </Button>
    </div>;

}

export default DetailPage;