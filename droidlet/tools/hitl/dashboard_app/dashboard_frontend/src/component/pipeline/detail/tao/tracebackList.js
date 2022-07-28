import React, { useContext, useEffect, useState } from "react";
import { SocketContext } from "../../../../context/socket";
import { Button, List, Table, Typography } from "antd";
import { CopyOutlined } from '@ant-design/icons';

const CopyButton = (props) => {
    const text = JSON.stringify(props.data);

    return <Button type="primary" icon={<CopyOutlined/>} onClick={() => { navigator.clipboard.writeText(JSON.stringify(text))}} />
}

const tracebackListCols = [
    {
        title: 'Chat Content',
        dataIndex: 'chat_content',
        render: (o) => (
            <>
                <List>
                    {
                        o.slice(0, 10).map((line) =>
                            line.length
                            &&
                            <List.Item>
                                <Typography.Paragraph ellipsis={{ rows: 2, expandable: true, symbol: 'more' }}>{line}</Typography.Paragraph>
                            </List.Item>
                        )
                    }
                    {
                        o.length > 10 &&
                        <List.Item>Copy All {o.length} Chat Contents to Clipboard.
                            <Button type="primary" icon={<CopyOutlined onClick={() => { navigator.clipboard.writeText(JSON.stringify(o))}} />} />
                        </List.Item>
                    }
                </List>

            </>
        )
    },
    {
        title: 'Content',
        dataIndex: 'content',
        sorter: (one, other) => (one.content.localeCompare(other.content)),
    },
    {
        title: 'Frequency',
        dataIndex: 'freq',
        sorter: (one, other) => (one.freq === other.freq ? 0 : (one.freq < other.freq ? 1 : -1))
    },
]

const TracebackList = (props) => {
    const socket = useContext(SocketContext);
    const batchId = props.batchId;
    const [listData, setListData] = useState([]);
    const [loading, setLoading] = useState(true);

    const handleReceivedTraceback = (data) => {
        if (data !== 404) {
            setListData(JSON.parse(data));
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
        Traceback List
        <Table
            columns={tracebackListCols}
            dataSource={listData}
            scroll={{ y: '80vh' }}
            loading={loading}
        />
    </div>;
}

export default TracebackList;