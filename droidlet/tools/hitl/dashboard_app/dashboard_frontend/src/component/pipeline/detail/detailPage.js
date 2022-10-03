/*
Copyright (c) Facebook, Inc. and its affiliates.

Detail page of a run.

Batach ID and pipeline type needs to be specified by the caller. 

Usage:
<DetailPage batchId={batchId} pipelineType={pipelineType} />
*/
import { Button, Collapse, Divider, Spin, Timeline, Typography } from "antd";
import React, { useCallback, useContext, useEffect, useState } from "react";
import {
    ClockCircleOutlined,
    CheckOutlined,
    LoadingOutlined,
    CaretRightOutlined
} from '@ant-design/icons';
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { toFirstCapital } from "../../../utils/textUtils";
import { TAB_ITEMS } from "../../../constants/pipelineConstants";
import MetaInfoDescription from "./metaInfoDescription";
import { SocketContext } from '../../../context/socket';
import { JOB_TYPES } from "../../../constants/runContants";
import ModelCard from "./asset/modelCard";
import DatasetCard from "./asset/datasetCard";

const { Title } = Typography;
const { Panel } = Collapse;

const DetailPage = (props) => {
    const socket = useContext(SocketContext);
    const pipelineType = props.pipelineType;
    const batch_id = useParams().batch_id;
    const [runInfo, setRunInfo] = useState(null);
    const [jobs, setJobs] = useState(null);
    const navigate = useNavigate();

    const handleReceivedRunInfo = useCallback((data) => {
        if (data === 404) {
            // received error code
            navigate("/notfound");
        }
        setRunInfo(data);
        setJobs(getJobs(data));
    }, []);

    const getRunInfo = () => {
        socket.emit("get_run_info_by_id", batch_id);
    }

    useEffect(() => getRunInfo(), []); // load job list when init the component (didMount)

    useEffect(() => {
        socket.on("get_run_info_by_id", (data) => handleReceivedRunInfo(data));
    }, [socket, handleReceivedRunInfo]);

    useEffect(() => { }, [runInfo]);

    const getJobs = (runInfo) => {
        const jobs = Object
            .entries(runInfo)
            .filter((obj) => (obj[0] in JOB_TYPES));
        return jobs;
    };

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
            return <Button type="primary" value={jobName}><Link to={jobKey.toLocaleLowerCase()}>{jobName}</Link></Button>;
        } else {
            return <Typography.Text>{jobName}</Typography.Text>;
        }
    }

    return <div style={{ 'padding': '0 24px 0 24px' }}>
        {runInfo ?
            <div>
                <Collapse
                    bordered={false}
                    expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
                    defaultActiveKey={['overview']}
                >
                    <Panel
                        header={<Title level={4}>{`Overview of Run ${batch_id}`}</Title>}
                        key='overview'
                    >
                        <MetaInfoDescription metaInfo={runInfo} />
                        <Divider />
                        <div style={{ 'display': 'flex'}}>
                            <DatasetCard batchId={batch_id} pipelineType={pipelineType} />
                            <ModelCard batchId={batch_id} pipelineType={pipelineType} />
                        </div>
                    </Panel>
                </Collapse>

                <Divider />
                <div style={{ 'display': 'flex', "padding": "0 32px 0 32px" }}>
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
                    <Outlet context={{ metaInfo: runInfo }} />
                </div>
                <div style={{ "paddingTop": "18px" }}>
                    <Button type="primary">
                        <Link to="../" state={{ label: TAB_ITEMS.RUNS.label, key: TAB_ITEMS.RUNS.key }}>
                            Back to View All
                        </Link>
                    </Button>
                </div>
            </div> :
            <Spin />
        }
    </div>;

}

export default DetailPage;