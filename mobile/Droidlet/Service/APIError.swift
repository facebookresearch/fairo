enum APIError: Error {
    case decodingError
    case httpError(Int)
    case unknown
}
