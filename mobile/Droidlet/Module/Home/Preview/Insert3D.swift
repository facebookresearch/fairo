
import Foundation
import SceneKit
import ModelIO
import SceneKit.ModelIO

public struct Insert3DViewer {
    public var x: CGFloat
    public var y: CGFloat
    public var background: NSObject
    public init() {
        x = 0;
        y = 0;
        background = UIColor.black;
    }
}

public struct Insert3DModel {
    public var mesh: URL
    public var material: String
    public var autoRotate: Bool
    public var rotationSpeed: TimeInterval
    public var fixed: Bool
    public init() {
        mesh = URL(fileURLWithPath: "")
        material = ""
        autoRotate = false
        fixed = false
        rotationSpeed = 0
    }
}


extension UIView {
    
    public func Insert3D(viewerSetup: Insert3DViewer, modelSetup: Insert3DModel) {
        
        let scene = SCNScene()
        
        let asset = MDLAsset(url: modelSetup.mesh)
        guard let object = asset.object(at: 0) as? MDLMesh else {
            return
        }
        print("Model Loaded!")
        
        // Create a material from the texture
        let scatteringFunction = MDLScatteringFunction()
        let material = MDLMaterial(name: "mat1", scatteringFunction: scatteringFunction)
        
        let meshUrl = Bundle.main.url(forResource: modelSetup.material, withExtension: "")
        material.setProperty(MDLMaterialProperty(name: modelSetup.material, semantic: .baseColor, url: meshUrl))
        
        // Apply the texture to every submesh of the asset
        for  submesh in object.submeshes!  {
            if let submesh = submesh as? MDLSubmesh {
                submesh.material = material
                print("Material loaded!")
            }
        }
        
        // Wrap the ModelIO object in a SceneKit object
        let modelNode = SCNNode(mdlObject: object)
        scene.rootNode.addChildNode(modelNode)
        
        if modelSetup.autoRotate == true {
            modelNode.runAction(SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: 2, z: 0, duration: modelSetup.rotationSpeed
                                                                          )))
        }
        
        let scnView = SCNView(frame: CGRect(x: viewerSetup.x, y: viewerSetup.y, width: bounds.width, height: bounds.height))
        //self.view.addSubview(scnView)
        self.addSubview(scnView)
        scnView.scene = scene
        
        //set up scene
        scnView.autoenablesDefaultLighting = false
        scnView.allowsCameraControl = !modelSetup.fixed
        scnView.scene = scene
        //scnView.backgroundColor = UIColor.clear
        scene.background.contents = viewerSetup.background
    }
}
