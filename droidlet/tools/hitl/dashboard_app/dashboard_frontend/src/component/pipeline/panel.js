/*
Copyright (c) Facebook, Inc. and its affiliates.

The panel for managing runs & single run of a specified pipeline. 

Pipeline type needs to be specified by the caller. 

Usage:
<PipelinePanel pipelineType={pipelineType} />
*/
import { Breadcrumb, Layout, Tabs } from 'antd';
import React, { useEffect, useState } from 'react';
import { Link, Outlet, useLocation, useParams } from 'react-router-dom';
import { TAB_ITEMS } from '../../constants/pipelineConstants';
import InfoBlock from './metaInfo/infoBlock';
import RunList from './metaInfo/runList';

const menuItems = Object.values(TAB_ITEMS);
const { TabPane } = Tabs;
const { Content } = Layout;

const PipelinePanel = (props) => {
    // get pipeline type, passed in as props
    const pipelineType = props.pipelineType;
    // get batch id from location - only used for detail page
    let { batch_id } = useParams();

    const location = useLocation();
    // get state label - only used when navigating back from detail page & set current label to intro if not navigating back from detail page
    const stateLabel = location.state ? location.state.label :  TAB_ITEMS.INTRO.label;
    const [currentLabel, setCurrentLabel] = useState(stateLabel);

    // get state key if there is any and set defult selected key
    const stateKey = location.state ? location.state.key :  TAB_ITEMS.INTRO.key;
    const [activeKey, setActiveKey] = useState(stateKey);

    // for nav between tabs
    const handleTabOnclick = (key) => {
        const label = menuItems.find((item) => (item.key === key)).label;
        setActiveKey(key);
        setCurrentLabel(label);
    }

    useEffect(() => {
        setCurrentLabel(stateLabel);
        setActiveKey(stateKey)
    }, [stateLabel, stateKey]);

    return (
        <Layout>
            <Content
                style={{
                    padding: '0 50px',
                }}
            >
                {/* breadcrumbs starts */}
                <Breadcrumb
                    style={{
                        margin: '16px 0',
                    }}
                >
                    <Breadcrumb.Item>HITL</Breadcrumb.Item>
                    <Breadcrumb.Item>{pipelineType.label}</Breadcrumb.Item>
                    {
                        /* if has batch id, show view jobs label & batch id number, 
                        otherwise show current label */
                        batch_id ?
                            (<>
                                <Breadcrumb.Item>
                                    <Link
                                        to={pipelineType.key}
                                        state={{ label: TAB_ITEMS.RUNS.label, key: TAB_ITEMS.RUNS.key }}
                                        replace={true}>
                                        {TAB_ITEMS.RUNS.label}
                                    </Link>
                                </Breadcrumb.Item>
                                <Breadcrumb.Item>{batch_id}</Breadcrumb.Item>
                            </>)
                            :
                            <Breadcrumb.Item>{currentLabel}</Breadcrumb.Item>
                    }
                </Breadcrumb>
                {/* breadcrumbs ends */}

                <Layout
                    style={{
                        padding: '24px 0',
                        background: 'white',
                    }}
                >
                    {/* tabs / detail page content starts */}
                    {
                        // display detail page if has batch id
                        batch_id ?
                            <Outlet /> // outlet defined at the router - will render detail page
                            :
                            (<Tabs
                                activeKey={activeKey}
                                defaultActiveKey={activeKey}
                                tabPosition="left"
                                onTabClick={(evt) => { handleTabOnclick(evt) }}
                            >
                                {
                                    menuItems.map((item) => (
                                        <TabPane tab={item.label} key={item.key}>
                                            {
                                                // render job list if key is jobs, otherwise render info block
                                                item.key === TAB_ITEMS.RUNS.key ?
                                                    <RunList pipelineType={pipelineType} /> :
                                                    <InfoBlock infoType={item.key} pipelineType={pipelineType} />
                                            }
                                        </TabPane>
                                    ))
                                }
                            </Tabs>)
                    }
                    {/* tabs / detail page ends */}
                </Layout>
            </Content>
        </Layout>

    );
};

export default PipelinePanel;