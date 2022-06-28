import { Breadcrumb, Layout, Tabs } from 'antd';
import React, { useState } from 'react';
import { Link, Navigate, NavLink, useLocation, useParams } from 'react-router-dom';
import { TAB_ITEMS } from '../../constants/pipelineConstants';
import { SUBPATHS } from '../../constants/subpaths';
import InfoBlock from './metaInfo/infoBlock';
import JobList from './metaInfo/jobList';

const menuItems = Object.values(TAB_ITEMS);
const { TabPane } = Tabs;
const { Content } = Layout;

const PipelinePanel = (props) => {
    // get pipeline type, passed in as props
    const pipelineType = props.pipelineType;
    // get batch id from location - only used for detail page
    let { batch_id } = useParams();
    // get state label & key - only used when navigating back from detail page
    const location = useLocation();
    const stateLabel = location.state ? location.state.label : null;
    const stateKey = location.state ? location.state.key : null;
    // set current label to intro if not navigating back from detail page
    const [currentLabel, setCurrentLabel] = useState(stateLabel ? stateLabel : TAB_ITEMS.INTRO.label);

    // for nav between tabs
    const handleTabOnclick = (key) => {
        const label = menuItems.find((item) => (item.key === key)).label;
        setCurrentLabel(label);
    }

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
                    <Breadcrumb.Item>{pipelineType}</Breadcrumb.Item>
                    {
                        /* if has batch id, show view jobs label & batch id number, 
                        otherwise show current label */
                        batch_id ?
                            (<>
                                <Breadcrumb.Item>
                                    <Link
                                        to={SUBPATHS.NLU.key}
                                        state={{ label: TAB_ITEMS.JOBS.label, key: TAB_ITEMS.JOBS.key }}
                                        replace={true}>
                                        {TAB_ITEMS.JOBS.label}
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
                            (<div>detail of {batch_id} </div>)
                            :
                            (<Tabs
                                defaultActiveKey={TAB_ITEMS.INTRO.key}
                                tabPosition="left"
                                onTabClick={(evt) => { handleTabOnclick(evt) }}
                            >
                                {
                                    menuItems.map((item) => (
                                        <TabPane tab={item.label} key={item.key}>
                                            {
                                                // render job list if key is jobs, otherwise render info block
                                                item.key === TAB_ITEMS.JOBS.key ?
                                                    <JobList pipelineType={pipelineType} /> :
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