import Foundation

enum DataStoreStringNames: String{
    case locationInfo = "locationInfo"
}

class DataStore: NSObject, NSCoding {
    
    //MARK: - Properties
    
    private var locationInfoList: Array<LocationInfo> = []
    
    //MARK: - Init
    
    /*/ init()
     Init for the class
     */
    override init() {
        super.init()
    }
    
    //MARK: - Helper
    
    func setLocationInfoList(list: Array<LocationInfo>){
        self.locationInfoList = list
    }

    func getLocationInfoList() -> Array<LocationInfo> {
        return self.locationInfoList
    }
    
    //MARK: - Encode
    func encode(with coder: NSCoder) {
        coder.encode(self.locationInfoList, forKey: DataStoreStringNames.locationInfo.rawValue)
    }
    
    //MARK: - Decode
    required init?(coder decoder: NSCoder) {
        super.init()
        self.locationInfoList = decoder.decodeObject(forKey: DataStoreStringNames.locationInfo.rawValue) as! Array<LocationInfo>
        
    }
    
}
