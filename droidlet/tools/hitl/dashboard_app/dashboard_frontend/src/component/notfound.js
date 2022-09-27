/*
Copyright (c) Facebook, Inc. and its affiliates.

Not found page.
*/
import { FrownTwoTone } from "@ant-design/icons";
import { Card, Layout, Typography } from "antd";
import { Content } from "antd/lib/layout/layout";
import React from "react";

const NotFoundPage = () => (
    <Layout>
        <Content
            style={{
                padding: '24px 50px 50px 50px',
            }}
        >
            <Card style={{ height: '60vh', position: 'relative' }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'center'
                }}>
                    <FrownTwoTone
                        twoToneColor="#8c8c8c"
                        style={{
                            fontSize: '50px',
                            top: '30%',
                            position: 'absolute',
                        }} />
                    <Typography.Text
                        style={{
                            fontSize: '15px',
                            top: '50%',
                            position: 'absolute',
                            alignContent: 'center'
                        }}
                    >
                        Sorry, we cannot find the resource you are looking for.
                    </Typography.Text>
                </div>
            </Card>
        </Content>
    </Layout>
);

export default NotFoundPage;