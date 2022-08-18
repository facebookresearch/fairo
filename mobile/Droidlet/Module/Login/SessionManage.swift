import Foundation
import SimpleKeychain

class SessionManage {
    static let shared = SessionManage()
    let keychain = A0SimpleKeychain(service: "Auth")

    var accessToken: String? {
        return self.keychain.string(forKey: "access_token")
    }
    
    func storeAccessToken(_ accessToken: String) {
        self.keychain.setString(accessToken, forKey: "access_token")
    }
    
    func logout() {
        self.keychain.clearAll()
    }
}
