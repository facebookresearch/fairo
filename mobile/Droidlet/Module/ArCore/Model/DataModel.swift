import Foundation
import ARKit

class DataModel {
    //MARK: - Properties
    
    static let dataModelSharedInstance = DataModel()
    var sceneView: ARSCNView?
    var locationDetailsSharedInstance: LocationDetails?
    var locationManagerSharedInstance: LocationManager?
    var ARSCNViewDelegateSharedInstance: ARSceneViewDelegate?
    var nodeManagerSharedInstance: NodeManager?
    var dataStoreManagerSharedInstance: DataStoreManager?

    //MARK: - Helper Functions
    
    private func initLocationManager(){
        self.locationManagerSharedInstance!.requestLocationAuth(success: {message in
        }, failure: {alertString in
        })
    }
    
    func initLocationDetails(){
        self.locationDetailsSharedInstance?.reset()
    }
    
    private func initARSCNViewDelegate(){
    }
    
    private func initNodeManager(){
        self.nodeManagerSharedInstance?.reset()
    }
    
    private func initDataStoreManager(){
    }

    func resetNavigationToRestingState(){
        getLocationDetails().reset()
        getNodeManager().reset()
        getARNSCNViewDelegate().reset()
    }
    
    //MARK: - Setter Functions
    
    /*/ setLocationManager(locationManager: LocationManager)
     Sets locationManagerSharedInstance to a LocationManager
     
     @param: locationManager - a LocationManager Object
     */
    func setLocationManager(locationManager: LocationManager){
        locationManagerSharedInstance = locationManager
        initLocationManager()
    }
    /*/ setLocationDetails(locationDetails: LocationDetails)
     Sets locationDetailsSharedInstance to a LocationDetails
     
     @param: locationDetails - a LocationDetails Object
     */
    func setLocationDetails(locationDetails: LocationDetails){
        locationDetailsSharedInstance = locationDetails
        initLocationDetails()
    }
    /*/ setDataStoreManager(dataStoreManager: DataStoreManager)
     Sets dataStoreManagerSharedInstance to a DataStoreManager
     
     @param: dataStoreManager - a DataStoreManager Object
     */
    func setDataStoreManager(dataStoreManager: DataStoreManager){
        dataStoreManagerSharedInstance = dataStoreManager
        initDataStoreManager()
    }
    /*/ setARSCNViewDelegate(ARSCNViewDelegate: ARSceneViewDelegate)
     Sets ARSCNViewDelegateSharedInstance to a ARSceneViewDelegate
     
     @param: ARSCNViewDelegate - a ARSceneViewDelegate Object
     */
    func setARSCNViewDelegate(ARSCNViewDelegate: ARSceneViewDelegate){
        ARSCNViewDelegateSharedInstance = ARSCNViewDelegate
        initARSCNViewDelegate()
    }
    /*/ setNodeManager(nodeManager: NodeManager)
     Sets nodeManagerSharedInstance to a NodeManager
     
     @param: nodeManager - a NodeManager Object
     */
    func setNodeManager(nodeManager: NodeManager){
        nodeManagerSharedInstance = nodeManager
        initNodeManager()
    }

    func setSceneView(view: ARSCNView) {
        self.sceneView = view
    }
    
    //MARK: - Getter Functions
    
    func getLocationManager() -> LocationManager{
        return self.locationManagerSharedInstance!
    }

    func getLocationDetails() -> LocationDetails{
        return self.locationDetailsSharedInstance!
    }

    func getDataStoreManager() -> DataStoreManager{
        return self.dataStoreManagerSharedInstance!
    }

    func getARNSCNViewDelegate() -> ARSceneViewDelegate{
        return self.ARSCNViewDelegateSharedInstance!
    }

    func getNodeManager() -> NodeManager{
        return self.nodeManagerSharedInstance!
    }

    func getSceneView() -> ARSCNView {
        return self.sceneView!
    }
}
