import React from "react";
import Paper from "@material-ui/core/Paper";
import Grid from "@material-ui/core/Grid";
import { Typography } from "@material-ui/core";

function DetailRow(props) {
  if (null === props.value || undefined === props.value) {
    return null;
  }
  return (
    <>
      <Grid item xs={4}>
        <Typography variant="button" align="right">
          {props.title}
        </Typography>
      </Grid>
      <Grid item xs={8}>
        <Typography variant="body1" align="left">
          {props.value}
        </Typography>
      </Grid>
    </>
  );
}

class MemoryDetail extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    const [x, y, getMemoryForUUID] = this.props.memoryManager;

    const memory = getMemoryForUUID(this.props.uuid);

    return (
      <Grid container spacing={2} style={{ padding: "16px" }}>
        <Grid item xs={12}>
          <Typography variant="h4" align="left">
            Attributes
          </Typography>
        </Grid>
        <DetailRow title="UUID" value={memory[0]} />
        <DetailRow title="ID" value={memory[1]} />
        <DetailRow title="X" value={memory[2]} />
        <DetailRow title="Y" value={memory[3]} />
        <DetailRow title="Z" value={memory[4]} />
        <DetailRow title="Yaw" value={memory[5]} />
        <DetailRow title="Pitch" value={memory[6]} />
        <DetailRow title="Type" value={memory[9]} />
        <DetailRow title="Name" value={memory[7]} />
        <Grid item xs={12}>
          <Typography variant="h4" align="left">
            Triples
          </Typography>
        </Grid>

        {memory.triples.map((triple) => {
          return (
            <>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  {triple[4]}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  {triple[6]}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  Confidence: {triple[7]}
                </Typography>
              </Grid>
            </>
          );
        })}
      </Grid>
    );
  }
}

export default MemoryDetail;
