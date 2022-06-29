/*
Copyright (c) Facebook, Inc. and its affiliates.

List view of all runs for the specified pipeline. 
Pipeline type needs to be specifed by adding the pipelineType props by the caller. 

To use this component:
<RunList pipelineType={pipelineType} />
*/
import { Badge, Skeleton, Table, Typography } from 'antd';
import React, { useContext, useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { SocketContext } from '../../../context/socket';

const { Title } = Typography;

const timeStrComp = (one, other) => {
    // compare two time (format in string)
    const one_dt = Date.parse(one);
    const other_dt = Date.parse(other);
    if (one_dt === other_dt) {
        return 0;
    } else {
        return one_dt < other_dt ? -1: 1;
    }
}

const runListCols = [
    {
        title: 'Name',
        dataIndex: 'name',
        sorter: (one, other) => (one.name.localeCompare(other.name)),
    }, {
        title: 'Batch ID',
        dataIndex: 'batch_id',
        sorter: (one, other) => (one.batch_id === other.batch_id ? 0 : (one.batch_id < other.batch_id ? -1 : 1)),
    }, {
        title: 'Status',
        key: 'status',
        filters: [
            {
                text: 'Finished',
                value: 'done',
            }, {
                text: 'Running',
                value: 'running',
            }
        ],
        onFilter: (val, row) => (row.status === val), 
        render: (_, row) => (
            <span>
                {row.status === 'done' ?
                    <>                    
                        <Badge color='green' />
                        Finished
                    </> :
                    <>                    
                        <Badge color='yellow' />
                        Running
                    </>
                }

            </span>
        ), 
        sorter: (one, other) => (one.status.localeCompare(other.status)),  
    }, { 
        title: 'Start Time',
        dataIndex: 'start_time',
        sorter: (one, other) => (timeStrComp(one.start_time, other.start_time)),
    }, {
        title: 'End Time',
        dataIndex: 'end_time',
        sorter: (one, other) => (timeStrComp(one.end_time, other.end_time)),
    }, {
        title: 'Description',
        dataIndex: 'description',
        sorter: (one, other) => {
            if (one.description && other.description) {
                return one.description.localeCompare(other.description);
            } else {
                return one.description ? -1 : 0;
            }
        },
        ellipsis: true
    }, {
        title: 'Action',
        key: 'action',
        render: (_, row) => (
            <Link to={`${row.batch_id}`}>View Detail</Link>
        )
    }
]

const RunList = (props) => {
    const socket = useContext(SocketContext);

    const pipelineType = props.pipelineType; // to do get runs by pipelineType

    const [runListData, setRunListData] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleReceivedRunList = useCallback((data) => {
        setRunListData(data.map((o) => (
            // TODO: update backend api for get job list, 
            // right now using fake name, descrioption infomation, and status
            {
                name: `name${o}`,
                batch_id: o,
                description: 'some description here',
                status: o % 2 === 0 ? 'done': 'running',
                start_time: `${o.toString().substring(0, 4)}-${o.toString().substring(4, 6)}-${o.toString().substring(6, 8)} 12:00:${o.toString().substring(12)}`,
                end_time: `${o.toString().substring(0, 4)}-${o.toString().substring(4, 6)}-${o.toString().substring(6, 8)} 18:30:${o.toString().substring(12)}`
            }
        )));
        setLoading(false);
    }, []);

    const getRunList = () => {
        socket.emit("get_job_list");
        setLoading(true);
    }

    useEffect(() => getRunList(), []); // load job list when init the component (didMount)

    useEffect(() => {
        socket.on("get_job_list", (data) => handleReceivedRunList(data));
    }, [socket, handleReceivedRunList]);

    return (
        <>
            <div style={{ "text-align": "left" }}>
                <Title level={5}>
                    View All {pipelineType.label} Runs
                </Title>
                {loading ?
                    <Skeleton active={true} title={false} paragraph={{ rows: 1, width: 200 }} />
                    :
                    <p>Showing {runListData.length} past runs.</p>}
            </div>
            <div style={{ "margin-right": "24px" }}>
                <Table
                    columns={runListCols}
                    dataSource={runListData}
                    scroll={{ y: '80vh' }}
                    loading={loading}
                />
            </div>
        </>
    );
}

export default RunList;