/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';

const KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

const INDEX_MAP =  {}
KEYPOINTS.map((item, index)=>{
    INDEX_MAP[item] = index
})

const DRAW_LINES = [
    ['left_ear', 'left_eye'],
    ['right_ear', 'right_eye'],
    ['left_eye', 'right_eye'],
    ['left_eye', 'nose'],
    ['right_eye', 'nose'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['left_shoulder', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['right_shoulder', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
    ['right_hip', 'left_hip'],
    ['right_shoulder', 'left_shoulder']
]

const PT_SIZE = 5

/*Props

imgUrl (URL): The location of the image to load into the tool

submitCallback (Function): A function to be called once a pose is completed

*/

class PoseTool extends React.Component {
    constructor(props){
        super(props)
        
        this.update = this.update.bind(this)
        this.onClick = this.onClick.bind(this)
        this.onMouseMove = this.onMouseMove.bind(this)
        this.keyDown = this.keyDown.bind(this)

        this.localToImage = this.localToImage.bind(this)
        this.imageToLocal = this.imageToLocal.bind(this)
        this.shiftViewBy = this.shiftViewBy.bind(this)

        this.dragging = false

        this.canvasRef = React.createRef()
        this.imgRef = React.createRef()

        this.points = []
        this.currentKeypointIndex = 0
        this.finished = false

        this.lastMouse = {x:0, y:0}
        this.Offset = {x:0,y:0}
    }

    componentDidMount(){
        this.canvas = this.canvasRef.current
        this.ctx = this.canvas.getContext('2d')

        this.img = this.imgRef.current
        this.img.onload = () => {
            this.baseScale = Math.min(this.canvas.width/this.img.width, this.canvas.height/this.img.height)
            this.scale = this.baseScale
            this.update()
        }
    }

    render(){
        let HeaderText = <p>Please click the {KEYPOINTS[this.currentKeypointIndex]}</p>
        if(this.dragging) {
            HeaderText = <p>Click to place the {KEYPOINTS[this.draggingIndex]}</p>
        } else if (this.finished) {
            HeaderText = <p>Press enter to submit</p>
        }

        return(
            <div>
                {HeaderText}
                <canvas ref={this.canvasRef} width="500px" height="500px" tabIndex="0"
                    onClick={this.onClick} 
                    onMouseMove={this.onMouseMove}
                    onKeyDown={this.keyDown}>
                </canvas>
                <img ref={this.imgRef} src={this.props.imgUrl} hidden={true}></img>
            </div>
        )
    }

    update(){
        //clear and transform
        this.ctx.resetTransform()
        this.ctx.clearRect(0,0,this.canvas.width, this.canvas.height)
        this.ctx.setTransform(this.scale, 0, 0, this.scale, this.Offset.x, this.Offset.y)
        //Draw image scaled and repostioned
        this.ctx.drawImage(this.img, 0, 0)
        //Draw points and lines
        this.points.map((pt)=>{
            this.drawPoint(pt)
        })
        this.drawPoseLines()

    }

    onMouseMove(e){
        var rect = this.canvas.getBoundingClientRect();
        this.lastMouse = {
            x: (e.clientX - rect.left),
            y: e.clientY - rect.top +1
        }
        if (this.dragging){
            this.points[this.draggingIndex] = this.localToImage(this.lastMouse)
        }
        this.update()
        this.forceUpdate()
    }

    onClick(e){
        if(this.dragging){
            this.dragging = false
            this.update()
            return
        }

        let hoverPointIndex = null
        for(let i = 0; i < this.points.length; i++){
            if(this.distance(this.points[i], this.localToImage(this.lastMouse)) < 10){
                hoverPointIndex = i
            }
        }

        if(hoverPointIndex != null){
            this.dragging = true
            this.draggingIndex = hoverPointIndex
            this.update()
            return
        }

        this.points.push(this.localToImage(this.lastMouse))
        this.currentKeypointIndex += 1
        if (this.currentKeypointIndex >= KEYPOINTS.length){
            this.finished = true
        }
        
        this.forceUpdate()
        this.update()
        this.lastKey = 'Mouse'
    }

    keyDown(e){
        switch(e.key){
            case(' '):
                if (this.points.length > 0){
                    this.points.pop()
                }
                break
            case('Enter'):
                if (this.finished) {
                    this.props.submitCallback(this.points)
                }
        }
        this.lastKey = e.key
        this.update()
    }

    drawPoint(pt){
        this.ctx.fillStyle = 'blue';
        if (this.distance(pt, this.localToImage(this.lastMouse)) < PT_SIZE*2){
            this.ctx.fillStyle = 'yellow';
        }
        this.ctx.fillRect(pt.x -PT_SIZE, pt.y -PT_SIZE, PT_SIZE*2, PT_SIZE*2);
    }

    drawLine(pt1, pt2, context=this.ctx, lineWidth=5){
        context.beginPath()
        context.moveTo(pt1.x, pt1.y)
        context.lineTo(pt2.x, pt2.y)
        context.lineWidth = lineWidth
        context.stroke()
    }

    drawPoseLines(){
        DRAW_LINES.map((pair)=>{
            let i0 = INDEX_MAP[pair[0]]
            let i1 = INDEX_MAP[pair[1]]
            if(Math.max(i0, i1) < this.currentKeypointIndex){
                this.drawLine(this.points[i0], this.points[i1])
            }
        })
    }

    drawCrosshairs(context){
        this.drawLine({x:50,y:0}, {x:50, y:100}, context, 2)
        this.drawLine({x:0,y:50}, {x:100, y:50}, context, 2)
    }

    localToImage(pt){
        return {
            x: (pt.x - this.Offset.x)/this.scale,
            y: (pt.y - this.Offset.y)/this.scale 
        }
    }

    imageToLocal(pt){
        return {
            x: pt.x * this.scale + this.Offset.x,
            y: pt.y * this.scale + this.Offset.y
        }
    }

    resetView(){
        this.Offset = {
            x: 0,
            y: 0
        }
        this.scale = Math.min(this.canvas.width/this.img.width, this.canvas.height/this.img.height)
        
    }

    shiftViewBy(x, y){
        this.Offset = {
            x: this.Offset.x + x,
            y: this.Offset.y + y
        }
    }

    distance(pt1, pt2){
        return Math.abs(pt1.x-pt2.x) + Math.abs(pt1.y - pt2.y)
    }
}

export default PoseTool