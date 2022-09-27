import Foundation
import CoreLocation
import ARKit

class LocationManager: NSObject, CLLocationManagerDelegate{
    
    static let locationManagerInstance = LocationManager()
    
    //MARK: - Properties
    
    let locationManager = CLLocationManager()
    var dataModelSharedInstance: DataModel?

    //MARK: - Init
    override init() {
        super.init()
        dataModelSharedInstance = DataModel.dataModelSharedInstance
    }
    //MARK: - Helper Functions
    
    func requestLocationAuth(success: @escaping((String) -> Void), failure: @escaping((String) -> Void)){
        if CLLocationManager.locationServicesEnabled() == true {
            if CLLocationManager.authorizationStatus() == .restricted || CLLocationManager.authorizationStatus() == .denied || CLLocationManager.authorizationStatus() == .notDetermined{
                failure("locationServiceError")
            } else {
                locationManager.requestWhenInUseAuthorization()
                locationManager.desiredAccuracy = kCLLocationAccuracyNearestTenMeters
            }
        } else {
            failure("locationServiceError")
        }
        success("locationSuccess")
        self.locationManager.delegate = self
        locationManager.startUpdatingLocation()
    }
    
}
