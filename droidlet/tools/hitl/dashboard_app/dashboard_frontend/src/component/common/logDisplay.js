import { List, Typography } from "antd";
import React from "react";

const { Text } = Typography;

const LogDisplayCmp = (props) => {
    const log = props.log;

    return <List size="small" bordered>
        {
            log.map((line, idx) => (
                <List.Item>
                    <div>
                        <span style={{display: 'inline-block', width: '48px'}}><Text strong>{`${idx}`}</Text></span>
                        <span>{line}</span>
                    </div>
                </List.Item>
            ))
        }
    </List>
}

export default LogDisplayCmp;