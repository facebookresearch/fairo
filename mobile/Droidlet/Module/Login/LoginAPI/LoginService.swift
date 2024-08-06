import Foundation
import Combine

class LoginService {
    static let shared = LoginService()
    func login(_ token: String) -> AnyPublisher<LoginModel, APIError> {
        guard let url = LoginEndpoint.login(token: token).urlRequest.url else {
            return Fail(error: APIError.unknown)
                .eraseToAnyPublisher()
        }

        let urlRequest = LoginEndpoint.login(token: token).urlRequest
        return URLSession.shared.dataTaskPublisher(for: urlRequest)
            .map(\.data)
            .decode(type: LoginModel.self, decoder: JSONDecoder())
            .mapError { _ in APIError.unknown }
            .eraseToAnyPublisher()
    }

}
