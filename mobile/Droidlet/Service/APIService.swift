import Foundation
import Combine

protocol APIService {
 func request<T: Decodable>(with builder: APIRequest) -> AnyPublisher<T, APIError>
}
