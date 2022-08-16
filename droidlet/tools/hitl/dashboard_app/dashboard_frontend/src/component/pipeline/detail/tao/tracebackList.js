/*
Copyright (c) Facebook, Inc. and its affiliates.

Traceback List for displaying traceback.

Usage:
<TracebackList />
*/

import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";
import { Table, Tag, Tooltip, Typography } from "antd";
import { MinusSquareTwoTone, PlusSquareTwoTone } from "@ant-design/icons";

const innerListCols = [
    // columns for list of each of the row in the traceback list
    {
        title: "Cause Command",
        dataIndex: "command",
        sorter: (one, other) => (one.command.localeCompare(other.command)),
        render: (_, row) => (
            <Typography.Paragraph>{row.command}</Typography.Paragraph>
        )
    }, {
        title: "Frequency",
        dataIndex: "freq",
        sorter: (one, other) => (one.freq === other.freq ? 0 : (one.freq < other.freq ? 1 : -1)),
        render: (_, row) => (<HeatColoredNum num={row.freq}/>),
    }
]

const tracebackListCols = [
    {
        title: "Traceback Content",
        dataIndex: "content",
        sorter: (one, other) => (one.content.localeCompare(other.content)),
        render: (_, row) => (
            <Typography.Paragraph
                style={{ whiteSpace: "pre-line" }}
            >
                {row.content}
            </Typography.Paragraph>)
    },
    {
        title: "Frequency",
        dataIndex: "freq",
        width: "10%",
        sorter: (one, other) => (one.freq === other.freq ? 0 : (one.freq < other.freq ? 1 : -1)),
        render: (_, row) => (<HeatColoredNum num={row.freq}/>),
    }, 
    {
        title: "Causes Count",
        dataIndex: "chat_content",
        width: "10%",
        render: (_, row) => (<HeatColoredNum num={row.chat_content.length}/>),
        sorter: (one, other) => (one.chat_content.length === other.chat_content.length ?
            0 :
            (one.chat_content.length < other.chat_content.length ? 1 : -1)
        )
    },
    Table.EXPAND_COLUMN
]

const HeatColoredNum = (props) => {
    const num = props.num;
    const colorList = ["magenta", "red", "volcano", "orange", "gold", "lime", "green", "cyan", "blue", "geekblue"]; // for color gradient for the tag

    return <Tag color={colorList[num > 100 ? 0 : Math.trunc(10 - num / 10)]}>{num}</Tag>
}

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
            scroll={{ y: "80vh" }}
            expandable={{
                // inner list, shows the command that causing the traceback
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