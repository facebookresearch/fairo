import { Button, Card, Typography } from "antd";
import React from "react";
import { Link, useOutletContext, useParams } from "react-router-dom";
import { toFirstCapital } from "../../../../utils/textUtils";

const { Title } = Typography;

const JobInfoCard = (props) => {
    const batchId = useParams().batch_id;
    const job = useParams().job;
    const metaInfo = useOutletContext().metaInfo;
    console.log(metaInfo);

    // TODO: add backend call
    // TODO: add log display
    // TODO: add job meta data description

    return <div style={{ 'paddingLeft': '12px' }}>
        <Card
            title={`${toFirstCapital(job)} Jobs`}
            extra={<Button type="link"><Link to="../">Close</Link></Button>}
        >

        </Card>
    </div>;
}
export default JobInfoCard;