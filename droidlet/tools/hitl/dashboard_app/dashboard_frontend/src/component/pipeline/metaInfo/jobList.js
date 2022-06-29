import { Badge, Skeleton, Table, Typography } from 'antd';
import React, { useContext, useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { SocketContext } from '../../../context/socket';

const { Title } = Typography;

const jobListCols = [
    {
        title: 'Name',
        dataIndex: 'name',
        sorter: (one, other) => (one.name.localeCompare(other.name)),
        width: '15%'
    }, {
        title: 'Batch ID',
        dataIndex: 'batch_id',
        sorter: (one, other) => (one.batch_id === other.batch_id ? 0 : (one.batch_id < other.batch_id ? -1 : 1)),
        width: '10%'
    }, {
        title: 'Status',
        key: 'status',
        width: '10%',
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
        width: '10%',
        render: (_, row) => (
            <Link to={`${row.batch_id}`}>View Detail</Link>
        )
    }
]

const JobList = (props) => {
    const socket = useContext(SocketContext);

    const pipelineType = props.pipelineType; // to do get jobs by pipelineType

    const [jobListData, setJobListData] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleReceivedJobList = useCallback((data) => {
        setJobListData(data.map((o) => (
            // TODO: update backend api for get job list, 
            // right now using fake name, descrioption infomation, and status
            {
                name: `name${o}`,
                batch_id: o,
                description: 'some description here',
                status: o % 2 === 0 ? 'done': 'running',
            }
        )));
        setLoading(false);
    }, []);

    const getJobList = () => {
        socket.emit("get_job_list");
        setLoading(true);
    }

    useEffect(() => getJobList(), []); // load job list when init the component (didMount)

    useEffect(() => {
        socket.on("get_job_list", (data) => handleReceivedJobList(data));
    }, [socket, handleReceivedJobList]);

    return (
        <>
            <div style={{ "text-align": "left" }}>
                <Title level={5}>
                    View All {pipelineType} Jobs
                </Title>
                {loading ?
                    <Skeleton active={true} title={false} paragraph={{ rows: 1, width: 200 }} />
                    :
                    <p>Showing {jobListData.length} past runs.</p>}
            </div>
            <div style={{ "margin-right": "24px" }}>
                <Table
                    columns={jobListCols}
                    dataSource={jobListData}
                    scroll={{ y: '80vh' }}
                    loading={loading}
                />
            </div>
        </>
    );
}

export default JobList;