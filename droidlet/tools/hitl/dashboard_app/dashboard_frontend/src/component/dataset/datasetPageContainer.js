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