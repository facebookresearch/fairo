import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";
import { Table, Tooltip, Typography } from "antd";
import { MinusSquareTwoTone, PlusSquareTwoTone } from '@ant-design/icons';

const innerListCols = [
    {
        title: 'Cause Command',
        dataIndex: 'command',
        sorter: (one, other) => (one.command.localeCompare(other.command)),
    }, {
        title: 'Frequency',
        dataIndex: 'freq',
        sorter: (one, other) => (one.freq === other.freq ? 0 : (one.freq < other.freq ? 1 : -1))
    }
]

const tracebackListCols = [
    {
        title: 'Traceback Content',
        dataIndex: 'content',
        sorter: (one, other) => (one.content.localeCompare(other.content)),
        render: (_, row) => (
            <Typography.Paragraph
                style={{ whiteSpace: "pre-line" }}
            >
                {row.content}
            </Typography.Paragraph>)
    },
    {
        title: 'Frequency',
        dataIndex: 'freq',
        sorter: (one, other) => (one.freq === other.freq ? 0 : (one.freq < other.freq ? 1 : -1))
    }
]

const TracebackList = (props) => {
    const socket = useContext(SocketContext);
    const batchId = props.batchId;
    const [listData, setListData] = useState([]);
    const [loading, setLoading] = useState(true);

    const handleReceivedTraceback = (data) => {
        if (data !== 404) {
            console.log(JSON.parse(data).map(
                (o) => ({
                    content: o.content,
                    freq: o.freq,
                    chat_content:
                        Object.keys(o.chat_content).map((k) => ({ command: k, freq: o.chat_content[k] }))
                })
            ))
            setListData(JSON.parse(data).map(
                (o) => ({
                    content: o.content,
                    freq: o.freq,
                    chat_content:
                        Object.keys(o.chat_content).map((k) => ({ command: k, freq: o.chat_content[k] }))
                })
            ));
        }
        setLoading(false);

    }

    useEffect(() => {
        socket.emit("get_traceback_by_id", batchId);
    }, []);

    useEffect(() => {
        socket.on("get_traceback_by_id", (data) => handleReceivedTraceback(data));
    }, [socket, handleReceivedTraceback]);

    return <div>
        <Table
            title={() => <Typography.Title level={4} style={{ textAlign: "left" }}>Tracebacks</Typography.Title>}
            columns={tracebackListCols}
            dataSource={listData}
            scroll={{ y: '80vh' }}
            expandable={{
                expandedRowRender: (row) =>
                    <Table
                        bordered
                        columns={innerListCols}
                        dataSource={row.chat_content}
                        pagination={row.chat_content.length > 10}
                    />,
                expandRowByClick: true,
                indentSize: 0,
                expandIcon: ({ expanded, onExpand, record }) =>
                    expanded ? (
                        <Tooltip title="Close"><MinusSquareTwoTone onClick={e => onExpand(record, e)} /></Tooltip>
                    ) : (
                        <Tooltip title="View Causes"><PlusSquareTwoTone onClick={e => onExpand(record, e)} /></Tooltip>
                    )
            }}
            loading={loading}
        />
    </div>;
}

export default TracebackList;