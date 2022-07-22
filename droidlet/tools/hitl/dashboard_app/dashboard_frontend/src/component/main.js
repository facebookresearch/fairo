/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import { Layout, Typography, Card, Divider } from 'antd';
import { SearchOutlined, ToolOutlined } from '@ant-design/icons';
import { Content } from 'antd/lib/layout/layout';

const Main = () => {
    return <div>
        <Layout>
            <Content style={{
                padding: '24px 50px 50px 50px',
            }}>
                <Typography.Title>Welcome to Droidlet HITL Dashboard.</Typography.Title>
                <Card style={{ height: '60vh', position: 'relative' }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'center'
                    }}>
                    </div>
                    <Divider>
                        <Typography.Title level={5}>
                            <SearchOutlined style={{ paddingRight: '4px' }} />
                            Quick View Run
                        </Typography.Title>
                    </Divider>

                    <Divider>
                        <Typography.Title level={5}>
                            <ToolOutlined style={{ paddingRight: '4px' }} />
                            Tools
                        </Typography.Title>
                    </Divider>
                    <Card
                        title="Checksum"
                        type="inner"
                    >
                        Checksum 

                    </Card>
                </Card>
            </Content>
        </Layout>
    </div>

}

export default Main;