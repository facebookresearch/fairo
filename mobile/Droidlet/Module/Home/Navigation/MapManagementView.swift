import SwiftUI

struct MapManagementView: View {
	var dataModelSharedInstance = DataModel.dataModelSharedInstance
	@ObservedObject var viewModel: HomeViewModel
	@Environment(\.presentationMode) var presentationMode
	@State private var showingOptions = false
	
	var body: some View {
		ZStack {
			VStack {
				HStack {
					Button("Close") {
						presentationMode.wrappedValue.dismiss()
					}
					.padding(.top, 10)
					.padding(.leading, 15)
					
					Spacer()
					
					Button("Menu") {
						showingOptions.toggle()
					}
					.padding(.top, 10)
					.padding(.trailing, 15)
				}
				List(dataModelSharedInstance.getDataStoreManager().dataStore.getLocationInfoList(), id: \.self) { location in
					Button(location.destination) {
						print("**** Selected map: \(location.destination)")
						
						presentationMode.wrappedValue.dismiss()
						viewModel.selectedLocation = location
						viewModel.arState = .navigation
						viewModel.navigationHintState = ""
						let nodeList = location.nodes.index
						viewModel.navigationHintState = "Scan the marker to navigate to: \(location.destination)"
						dataModelSharedInstance.getNodeManager().setNodeList(list: nodeList)
						dataModelSharedInstance.getNodeManager().setIsNodeListGenerated(isSet: true)
						dataModelSharedInstance.getLocationDetails().setIsNavigating(isNav: true)
						viewModel.resetToNormalState()
					}
					.frame(maxWidth: .infinity, maxHeight: 50)
				}
				.background(Color.yellow)
			}
		}
		.actionSheet(isPresented: $showingOptions) {
			ActionSheet(
				title: Text("What would you like to do?"),
				buttons: [
					.default(Text("Create Custom Map")) {
						LocationDetails.LocationDetailsSharedInstance.setIsCreatingCustomMap(isCreatingCustomMap: true)
						viewModel.arState = .findingMarker
						viewModel.navigationHintState = "Scan the marker to start!"
						presentationMode.wrappedValue.dismiss()
						viewModel.resetToNormalState()
					},
						.default(Text("Upload Custom Map")) {
							viewModel.uploadMap()
						},
						.cancel(Text("Cancel"), action: {
							showingOptions = false
						})
				]
			)
		}
	}
}

#if DEBUG
struct MapManagementView_Previews: PreviewProvider {
	static var previews: some View {
		MapManagementView(viewModel: HomeViewModel())
	}
}
#endif
