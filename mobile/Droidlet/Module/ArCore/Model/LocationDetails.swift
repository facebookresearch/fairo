import Foundation
import CoreLocation

import SceneKit
class LocationDetails{
    static let LocationDetailsSharedInstance = LocationDetails()
    
    //MARK: - Properties
    
    var sourcePosition: CLLocationCoordinate2D?
    var destinationPosition: CLLocationCoordinate2D?
    var userPositionWhenFoundBeacon: SCNVector3?
    var isNavigating: Bool!
    var isBeaconRootNodeFound: Bool!
    var destination: String!
    var isCreatingCustomMap: Bool!
    var isWorldOriginSet: Bool!
    var isUploadingMap: Bool!
    
    var dataModelSharedInstance: DataModel?
    //MARK: - Init
    private init() {
        self.dataModelSharedInstance = DataModel.dataModelSharedInstance
        reset()
    }
    //MARK: - Helper Functions
    
    /*/ reset()
     Resets data to resting state
     */
    func reset(){
        isNavigating = false
        isBeaconRootNodeFound = false
        destination = nil
        isCreatingCustomMap = false
        isWorldOriginSet = false
        isUploadingMap = false
    }
    
    //MARK: Setter Functions
    
    /*/ setIsNavigating(isNav: Bool)
     Sets isNavigating to a boolean value.
     
     @param: isNav - Bool - boolean value
     */
    func setIsNavigating(isNav: Bool){
        self.isNavigating = isNav
    }

    func setIsCreatingCustomMap(isCreatingCustomMap: Bool){
        self.isCreatingCustomMap = isCreatingCustomMap
    }

    func setIsBeaconRootNodeFound(isFound: Bool){
        self.isBeaconRootNodeFound = isFound
    }

    func setDestination(destination: String){
        self.destination = destination
    }
    /*/ setUserPositionWhenFoundBeacon (pos: SCNVector3)
     Sets userPositionWhenFoundBeacon to a SCNVector3 value.
     
     @param: userPositionWhenFoundBeacon - SCNVector3 - SCNVector3 value
     */
    func setUserPositionWhenFoundBeacon (pos: SCNVector3){
        self.userPositionWhenFoundBeacon = pos
    }
    /*/ setIsWorldOriginSet (isSet: Bool)
     Sets isWorldOriginSet to a Boolean value.
     
     @param: isWorldOriginSet - Bool - Boolean value
     */
    func setIsWorldOriginSet (isSet: Bool){
        self.isWorldOriginSet = isSet
    }
    /*/ setIsUploadingMap (isSet: Bool)
     Sets isUploadingMap to a Boolean value.
     
     @param: isSet - Bool - Boolean value
     */
    func setIsUploadingMap (isSet: Bool){
        self.isUploadingMap = isSet
    }
    
    //MARK: Getter Functions
    
    /*/ getIsNavigating() -> Bool
     returns isNavigating
     
     @return isNavigating - Bool - Boolean Value
     */
    func getIsNavigating() -> Bool {
        return isNavigating
    }
    /*/ getIsCreatingCustomMap() -> Bool
     returns isCreatingCustomMap
     
     @return isCreatingCustomMap - Bool - Boolean Value
     */
    func getIsCreatingCustomMap() -> Bool{
        return self.isCreatingCustomMap
    }
    /*/ getIsBeaconRootNodeFound() -> Bool
     returns isBeaconRootNodeFound
     
     @return isBeaconRootNodeFound - Bool - Boolean Value
     */
    func getIsBeaconRootNodeFound() -> Bool {
        return isBeaconRootNodeFound
    }
    /*/ getDestination() -> String
     returns destination
     
     @return destination - String - String Value
     */
    func getDestination() -> String{
        return self.destination
    }
    
    /*/ getUserPositionWhenFoundBeacon() -> SCNVector3
     Returns the last known user position at the moment the last beacon was scanned
     
     @return: userPositionWhenFoundBeacon - SCNVector3 - The user's position
     */
    func getUserPositionWhenFoundBeacon() -> SCNVector3 {
        return self.userPositionWhenFoundBeacon!
    }
    /*/ getIsWorldOriginSet() -> Bool
     Returns boolean stating whether or not the world origin is set with the ARConfiguration
     
     @return: isWorldOriginSet - Bool - Boolean value
     */
    func getIsWorldOriginSet() -> Bool {
        return isWorldOriginSet
    }
    /*/ getIsUploadingMap() -> Bool
     Returns boolean stating whether or not the app is uploading a map
     
     @return: isUploadingMap - Bool - Boolean value
     */
    func getIsUploadingMap() -> Bool {
        return isUploadingMap
    }
    
}
