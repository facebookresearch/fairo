/*
Copyright (c) Facebook, Inc. and its affiliates.

Main page of the HiTL dashboard app.
*/
import React, { useContext, useEffect, useState } from "react";
import { Layout, Typography, Card, Divider, Button, Descriptions, Spin } from "antd";
import { SearchOutlined, ToolOutlined } from "@ant-design/icons";
import { Content } from "antd/lib/layout/layout";
import { SocketContext } from "../context/socket";

const MODEL_CHECKSUM_TABS = [
    {
        key: "nlu",
        tab: "NLU",
        agents: [
            {
                label: "",
                key: null
            }
        ]
    },
    {
        key: "perception",
        tab: "Perception",
        agents: [
            {
                label: "Locaobot",
                key: "locobot"
            },
            {
                label: "Craftassist",
                key: "craftassist"
            }
        ]
    },
]

const ChecksumDescItem = (props) => {
    /**
     * Checksum Description Item Content
     * Used for each of modelName - agent pair
     */
    const modelName = props.modelName;
    const agent = props.agent.key;
    const socket = useContext(SocketContext);
    const [checksumData, setChecksumData] = useState(null);

    const handleReceivedChecksum = (data) => {
        if (data[2] === 404) {
            // received error code
            setChecksumData("NA");
        } else if (data[0] === modelName && data[1] === agent) {
            setChecksumData(data[2]);
        }
    }

    const handleUpdateRes = (data) => {
        if (data !== 200) {
            // failed, show alert
            alert("Update checksum failed, please try again later");
        } 
        // reterive new checksum
        !checksumData && socket.emit("get_model_checksum_by_name_n_agent", modelName, agent);
    }

    useEffect(() => {
        setChecksumData(null);
        socket.emit("get_model_checksum_by_name_n_agent", modelName, agent);
    }, [modelName, agent]);

    useEffect(() => {
        socket.on("get_model_checksum_by_name_n_agent", (data) => handleReceivedChecksum(data));
        socket.on("update_model_checksum_by_name_n_agent", (data) => handleUpdateRes(data));
    }, [handleReceivedChecksum, handleUpdateRes]);

    const handleOnClick = () => {
        setChecksumData(null);
        socket.emit("update_model_checksum_by_name_n_agent", modelName, agent);
    }

    return <div style={{ display: "flex" }}>
        <div>{checksumData ? checksumData: <Spin />}</div>
        <Button type="primary" size="small" style={{ marginLeft: "12px" }} onClick={() => handleOnClick()}>
            Compute Checksum
        </Button>
    </div>;
}

const ChecksumCardConetent = (props) => {
    /**
     * Content of the compute checksum card
     */
    const modelName = props.activeKey;
    const agents = MODEL_CHECKSUM_TABS.find((tab) => (tab.key === props.activeKey)).agents;

    return <div>
        {
            agents && <Descriptions bordered>
                {
                    agents.map((agent) =>
                        <Descriptions.Item label={`${agent.label} Checksum`}>
                            <ChecksumDescItem modelName={modelName} agent={agent} />
                        </Descriptions.Item>
                    )
                }
            </Descriptions>
        }
    </div>
}

const ComputeModelChecksumCard = () => {
    /**
     * Compute Model Checksum Card in the tool block
     */
    const [activeTabKey, setActiveKey] = useState(MODEL_CHECKSUM_TABS[0].key);

    const onTabChange = (key) => {
        setActiveKey(key);
    }

    return <Card
        type="inner"
        title={"Model Checksum"}
        tabList={MODEL_CHECKSUM_TABS}
        activeTabKey={activeTabKey}
        onTabChange={(key) => { onTabChange(key) }}
    >
        {activeTabKey && <ChecksumCardConetent activeKey={activeTabKey} />}
    </Card>
}

const Main = () => {
    return <div>
        <Layout>
            <Content style={{
                padding: "24px 50px 50px 50px",
            }}>
                <Typography.Title>Welcome to Droidlet HITL Dashboard.</Typography.Title>
                <Card style={{ height: "60vh", position: "relative" }}>
                    <div style={{
                        display: "flex",
                        justifyContent: "center"
                    }}>
                    </div>

                    {/* quick view block TODO */}
                    <Divider>
                        <Typography.Title level={5}>
                            <SearchOutlined style={{ paddingRight: "4px" }} />
                            Quick View Run
                        </Typography.Title>
                    </Divider>

                    {/* tool block */}
                    <Divider>
                        <Typography.Title level={5}>
                            <ToolOutlined style={{ paddingRight: "4px" }} />
                            Tools
                        </Typography.Title>
                    </Divider>
                    <ComputeModelChecksumCard />
                </Card>
            </Content>
        </Layout>
    </div>

}

export default Main;