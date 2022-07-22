/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React, { useState } from "react";
import { Layout, Typography, Card, Divider, Button, Descriptions } from "antd";
import { SearchOutlined, ToolOutlined } from "@ant-design/icons";
import { Content } from "antd/lib/layout/layout";

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

const ChecksumCardConetent = (props) => {
    const modelName = props.activeKey;
    const agents = MODEL_CHECKSUM_TABS.find((tab) => (tab.key === props.activeKey)).agents;

    const handleOnClick = (agent) => {
        console.log(agent);
    }

    return <div>
        {
            agents && <Descriptions bordered>
                {
                    agents.map((agent) =>
                        <Descriptions.Item label={`${agent.label} Checksum`}>
                            <div style={{display: "flex"}}>
                                <div>NA</div>
                                <Button type="primary" size="small" style={{marginLeft: "12px"}} onClick={() => handleOnClick(agent)}>
                                    Compute Checksum
                                </Button>
                            </div>
                        </Descriptions.Item>
                    )
                }
            </Descriptions>
        }
    </div>
}

const ComputeModelChecksumCard = () => {
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
                    <Divider>
                        <Typography.Title level={5}>
                            <SearchOutlined style={{ paddingRight: "4px" }} />
                            Quick View Run
                        </Typography.Title>
                    </Divider>

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