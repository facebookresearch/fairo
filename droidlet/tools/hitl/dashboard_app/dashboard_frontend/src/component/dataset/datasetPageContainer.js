/*
Copyright (c) Facebook, Inc. and its affiliates.

The container for dataset detail page. 
Renders either the datasetDetailPage or a selector for navigate to the detail page:
 - Renders the dataset path is dataset/:pipelineType format.
 - Renders a selector to allow redirect to specific dataset detail page when no piplineType specified.

Usage:
<DatasetPageContainer />
*/
import Layout, { Content } from "antd/lib/layout/layout";
import React from "react";
import { Outlet, useParams } from "react-router-dom";
import DatasetSelector from "./datasetSelector";

const DatasetPageContainer = (props) => {
    const pipelineParams = useParams(); //used to check if we are on a specific pipeline page

    return <Layout>
        <Content
            style={{
                padding: '24px 50px',
            }}
        >
            <Layout
                style={{
                    padding: '24px 0',
                    background: 'white',
                }}
            >
                {!Object.keys(pipelineParams).length ? <DatasetSelector /> : <Outlet />}
            </Layout>
        </Content>
    </Layout>;
}

export default DatasetPageContainer;