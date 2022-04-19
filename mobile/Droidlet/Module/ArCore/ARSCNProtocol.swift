import Foundation

protocol ARScnViewDelegate{
    func destinationReached()
    func toggleUndoButton(shouldShow: Bool)
    func toggleEndButton(shouldShow: Bool)
    func toggleSaveButton(shouldShow: Bool)
    func toggleAddButton(shouldShow: Bool)
	func toggleScanToNavigate(shouldShow: Bool)
}
