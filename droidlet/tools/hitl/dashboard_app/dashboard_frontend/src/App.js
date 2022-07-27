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
import { PIPELINES, SUBPATHS } from './constants/subpaths';
import { BackTop, Layout } from 'antd';
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
                <Route path={SUBPATHS.HOME.key} element={<Main />} />
                {PIPELINES.map((pipeline) =>
                  <Route path={pipeline.key} element={<PipelinePanel pipelineType={pipeline} />}>
                    <Route path=":batch_id" element={<DetailPage pipelineType={pipeline.label} />}>
                      <Route path=":job" element={<JobInfoCard />} />
                    </Route>
                  </Route>
                )}
                <Route path={SUBPATHS.OTHER.key} element={<div>Something else here</div>} />
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
