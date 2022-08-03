/*
Copyright (c) Facebook, Inc. and its affiliates.

Dashboard App root.

App top level shared componets and Routers are specified in this file. 
*/
import './App.css';
import 'antd/dist/antd.css';
import './index.css'
import React from 'react';
import { SocketContext, socket } from './context/socket';
import Main from './component/main';
import { Routes, Route, BrowserRouter } from "react-router-dom";
import NavBar from './component/navbar';
import { SUBPATHS } from './constants/subpaths';
import { BackTop, Layout, Typography } from 'antd';
import PipelinePanel from './component/pipeline/panel';
import DetailPage from './component/pipeline/detail/detailPage';
import JobInfoCard from './component/pipeline/detail/job/jobInfoCard';
import NotFoundPage from './component/notfound';
import DatasetPageContainer from './component/dataset/datasetPageContainer';
import DatasetDetailPage from './component/dataset/datasetDetailPage';

const { Header, Footer } = Layout;

function App() {
  return (
    <SocketContext.Provider value={socket}>
      <BrowserRouter>
        <div className="App">
          <Layout>
            <Header className="header">
              <NavBar />
            </Header>
            {/* Routes for different pipeline */}
            <Layout>
              <Routes>
                <Route path={SUBPATHS.HOME.key} element={<div><Typography.Title>Welcome to Droidlet HiTL Dashboard</Typography.Title></div>} />
                <Route path={SUBPATHS.NLU.key} element={<PipelinePanel pipelineType={SUBPATHS.NLU} />}>
                  <Route path=":batch_id" element={<DetailPage pipelineType={SUBPATHS.NLU.label} />}>
                    <Route path=":job" element={<JobInfoCard />} />
                  </Route>
                </Route>
                <Route path="dataset" element={<DatasetPageContainer />}> 
                  <Route path=":pipeline" element={<DatasetDetailPage />} />
                </Route>
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
              <BackTop />
            </Layout>
            <Footer
              style={{
                textAlign: 'center',
              }}
            >
              Droidlet HITL Dashboard ©2022
            </Footer>
          </Layout>
        </div>
      </BrowserRouter>
    </SocketContext.Provider>

  );
}

export default App;
