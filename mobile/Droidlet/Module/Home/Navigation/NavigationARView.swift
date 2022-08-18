import SwiftUI
import SceneKit
import ARKit
import CoreLocation

struct NavigationARView: View {
    var dataModelSharedInstance = DataModel.dataModelSharedInstance
	@ObservedObject var viewModel: HomeViewModel
    @Environment(\.presentationMode) var presentationMode
	var dataModelInstance: DataModel!
	
	var body: some View {
            ZStack {
                DroidletARViewContainer(viewModel: viewModel, isHome: false)
                    .edgesIgnoringSafeArea(.all)
                VStack {
                    HStack {
                        Button(action: {
                            self.presentationMode.wrappedValue.dismiss()
                        }) {
                            HStack(alignment: .center, spacing: 5.0) {
                                Image("ic-back")
                                    .renderingMode(.template)
                                    .resizable()
                                    .frame(width: 25, height: 25, alignment: .center)
                                    .foregroundColor(Color.accentColor.opacity(0.8))
                                    .allowsHitTesting(false)
                            }
                        }
                        .frame(width: 60, height: 50, alignment: .center)

                        Spacer()
                        
                        Button ("Manage Maps") {
                            viewModel.showManageMap.toggle()
                        }
                        .buttonStyle(AccentFilledButton())
                    }
                    .padding(.leading, 8)
                    .padding(.trailing, 8)

                    Spacer()
                }
                
                if viewModel.arState != .idle {
                    VStack {
                        HStack {
                            Spacer()
                            Button ("Cancel") {
                                viewModel.cancelNavigation()
                            }
                            .buttonStyle(AccentFilledButton())
                            .padding(.top, 80)
                            .padding(.trailing, 20)
                        }
                        Spacer()
                    }
                    
                    VStack {
                        Spacer()
                        HStack {
                            Text(viewModel.navigationHintState)
                                .frame(maxWidth: .infinity, minHeight: 35)
                                .padding()
                                .background(Color.green.opacity(0.6))
                                .foregroundColor(.white)
                                .font(.headline)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(.bottom, 123)
                        .frame(maxWidth: .infinity)
                    }
                    .frame(maxWidth: .infinity)
                }
                
                if viewModel.arState == .start {
                    VStack {
                        Spacer()
                        HStack {
                            Button ("Add") {
                                viewModel.handleAddButton()
                            }
                            .buttonStyle(AccentFilledButton())
                        }
                        .padding(.bottom, 20)
                    }
                }
                
                if viewModel.arState == .waypoint {
                    VStack {
                        Spacer()
                        HStack (spacing: 20){
                            Button ("Add destination") {
                                viewModel.handleEndButton()
                            }
                            .buttonStyle(AccentFilledButton())
                            
                            Button ("Add waypoint") {
                                viewModel.handleAddButton()
                            }
                            .buttonStyle(AccentFilledButton())
                        }
                        .padding(.bottom, 20)
                    }
                }
                
                if viewModel.arState == .destination {
                    VStack {
                        Spacer()
                        HStack {
                            Button (viewModel.mapSavingState.rawValue) {
                                switch viewModel.mapSavingState {
                                case .selectImage:
                                    viewModel.source = .camera
                                    viewModel.showPhotoPicker()
                                    
                                case .saveAndUpload:
                                    viewModel.saveDataStore()
                                }
                            }
                            .buttonStyle(AccentFilledButton())
                            .padding(.top, 20)
                        }
                        .padding(.bottom, 20)
                    }
                }
            }
		.popover(isPresented: $viewModel.showManageMap) {
			MapManagementView(viewModel: viewModel)
		}
        .sheet(isPresented: $viewModel.showPicker) {
            ImagePicker(sourceType: .camera) { image in
                viewModel.saveMapName(with: image)
            }
        }
        .onAppear() {
            if viewModel.didSelectNavigating {
                viewModel.didSelectNavigating = false
                self.showDirection()
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: {
                    viewModel.startARSection()
                })
            }
        }
        .onDisappear() {
            viewModel.pauseARSection()
        }
	}
    
    func showDirection() {
        guard let location = viewModel.selectedLocation else { return }
        viewModel.arState = .navigation
        viewModel.navigationHintState = ""
        let nodeList = location.nodes.index
        viewModel.navigationHintState = "Scan the marker to navigate to: \(location.destination)"
        dataModelSharedInstance.getNodeManager().setNodeList(list: nodeList)
        dataModelSharedInstance.getNodeManager().setIsNodeListGenerated(isSet: true)
        dataModelSharedInstance.getLocationDetails().setIsNavigating(isNav: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: {
            viewModel.resetToNormalState()
        })
    }
}
