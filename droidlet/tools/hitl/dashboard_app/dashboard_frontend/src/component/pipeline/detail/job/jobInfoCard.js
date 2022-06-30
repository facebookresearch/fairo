import { Button, Card, Descriptions } from "antd";
import React from "react";
import { Link, useOutletContext, useParams } from "react-router-dom";
import { toFirstCapital } from "../../../../utils/textUtils";

const JobInfoCard = (props) => {
    const batchId = useParams().batch_id;
    const job = useParams().job;
    const jobInfo = Object.entries(useOutletContext().metaInfo)
        .filter((o) =>
            o[0].toLowerCase() === job)[0][1];
    console.log(jobInfo)

    // TODO: add backend call
    // TODO: add log display
    // TODO: add job meta data description

    return <div style={{ 'paddingLeft': '12px' }}>
        <Card
            title={`${toFirstCapital(job)} Jobs`}
            extra={<Button type="link"><Link to="../">Close</Link></Button>}
        >
            <Descriptions 
            >
                {Object.entries(jobInfo).map(
                    (o) => (
                        <Descriptions.Item label={o[0]} >
                            {o[1]}
                        </Descriptions.Item>
                    )
                )}
            </Descriptions>

        </Card>
    </div>;
}
export default JobInfoCard;