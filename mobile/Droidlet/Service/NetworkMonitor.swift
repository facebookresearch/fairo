import Network

enum Reachable {
    case yes, no
}

enum Connection {
    case cellular, loopback, wifi, wiredEthernet, other
}

class NetworkMonitor {

    private let monitor: NWPathMonitor =  NWPathMonitor()

    init() {
        let queue = DispatchQueue.global(qos: .background)
        monitor.start(queue: queue)
    }
}

extension NetworkMonitor {
    func startMonitoring(callBack: @escaping (_ connection: Connection, _ rechable: Reachable) -> Void) -> Void {
        monitor.pathUpdateHandler = { path in

            let reachable = (path.status == .unsatisfied || path.status == .requiresConnection)  ? Reachable.no  : Reachable.yes

            if path.availableInterfaces.count == 0 {
                return callBack(.other, .no)
            } else if path.usesInterfaceType(.wifi) {
                return callBack(.wifi, reachable)
            } else if path.usesInterfaceType(.cellular) {
                return callBack(.cellular, reachable)
            } else if path.usesInterfaceType(.loopback) {
                return callBack(.loopback, reachable)
            } else if path.usesInterfaceType(.wiredEthernet) {
                return callBack(.wiredEthernet, reachable)
            } else if path.usesInterfaceType(.other) {
                return callBack(.other, reachable)
            }
        }
    }
}

extension NetworkMonitor {
    func cancel() {
        monitor.cancel()
    }
}
