/*
Copyright (c) Facebook, Inc. and its affiliates.

List view of all runs for the specified pipeline. 
Pipeline type needs to be specifed by adding the pipelineType props by the caller. 

To use this component:
<RunList pipelineType={pipelineType} />
*/
import { Badge, Button, DatePicker, Input, Select, Skeleton, Table, Typography } from 'antd';
import React, { useContext, useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { SocketContext } from '../../../context/socket';
import { FilterOutlined } from '@ant-design/icons';

const { Title } = Typography;
const { Search } = Input;
const { RangePicker } = DatePicker;
const { Option } = Select;

const timeStrComp = (one, other) => {
    // compare two time (format in string)
    const one_dt = Date.parse(one);
    const other_dt = Date.parse(other);
    if (one_dt === other_dt) {
        return 0;
    } else {
        return one_dt < other_dt ? -1 : 1;
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
    const [displayData, setDisplayData] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleReceivedRunList = useCallback((data) => {
        data = data.map((o) => (
            // TODO: update backend api for get job list, 
            // right now using fake name, descrioption infomation, and status
            {
                name: `name${o}`,
                batch_id: o,
                description: 'some description here',
                status: o % 2 === 0 ? 'done' : 'running',
                start_time: `${o.toString().substring(0, 4)}-${o.toString().substring(4, 6)}-${o.toString().substring(6, 8)} 12:00:${o.toString().substring(12)}`,
                end_time: `${o.toString().substring(0, 4)}-${o.toString().substring(4, 6)}-${o.toString().substring(6, 8)} 18:30:${o.toString().substring(12)}`
            }
        ));
        setRunListData(data);
        setDisplayData(data);
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

    const onSearch = (value) => {
        if (value) {
            setDisplayData(runListData.filter((o) =>
            (o.name.includes(value)
                || o.batch_id === value
                || o.description.includes(value)
            )));
        } else {
            setDisplayData(runListData);
        }
    }

    const onOk = (value) => {
        console.log('onOk: ', value);
    };

    return (
        <>
            <div style={{ "text-align": "left" }}>
                <Title level={5}>
                    View All {pipelineType.label} Runs
                </Title>

                <div style={{ 'display': 'flex', 'padding': '6px 0 12px 0' }}>
                    {/* filter & search component */}
                    <Search placeholder="Search by Name /Batch id/Description" allowClear onSearch={onSearch} enterButton />
                    <Input.Group compact>
                        <Select defaultValue="1">
                            <Option value="1">Filter Start Time</Option>
                            <Option value="2">Filter End Time</Option>
                        </Select>
                        <RangePicker
                            showTime={{
                                format: 'HH:mm',
                            }}
                            format="YYYY-MM-DD HH:mm"
                            onOk={onOk}
                        />
                        <Button 
                            type="primary" 
                            onClick={(o) => console.log(111)} 
                            icon={<FilterOutlined />}
                        />
                    </Input.Group>
                </div>

                {loading ?
                    <Skeleton active={true} title={false} paragraph={{ rows: 1, width: 200 }} />
                    :
                    <p>Showing {displayData.length} past runs.</p>}
            </div>
            <div style={{ "margin-right": "24px" }}>
                <Table
                    columns={runListCols}
                    dataSource={displayData}
                    scroll={{ y: '80vh' }}
                    loading={loading}
                />

            </div>
        </>
    );
}

export default RunList;