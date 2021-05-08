from torch import nn
import torch

class Unstructured3D(nn.Module):

    def __init__(self, dimensions=3, hidden_size=40, num_layers=4):
        super().__init__()
        # store hyperparameters
        self.dimensions = dimensions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # activation function
        self.lr = nn.LeakyReLU(0.1)
        self.sig = nn.Sigmoid()

        # model parameters
        assert(num_layers >= 2)
        first_layer = nn.Linear(dimensions, hidden_size)
        last_layer = nn.Linear(hidden_size, 1)
        middle_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)]
        self.linear_layers = nn.ModuleList([first_layer] + middle_layers + [last_layer])
        
    
    def forward(self, query_points):
        output = query_points

        # pass through each layer
        for i in range(self.num_layers):
            output = self.linear_layers[i](output)
            # LeakyReLU activation
            if i < self.num_layers - 1:
                output = self.lr(output)
            # Sigmoid for last layer
            else:
                output = self.sig(output)
        
        return output


class Unstructured2D(nn.Module):
    '''
    Learns occupancy as a function of camera parameters and 2d position
    Input parameters are just the extrinsics, 3x3 rotation matrix + 3 camera center
    '''

    def __init__(self, hidden_size=80, num_layers=8, num_cam_params=12):
        super().__init__()
        # store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_cam_params = num_cam_params

        # activation function
        self.lr = nn.LeakyReLU(0.1)
        self.sig = nn.Sigmoid()

        # other hyperparameters
        self.cam_feats = 20

        # model parameters
        assert(num_layers >= 2)
        self.cam_params_preprocess1 = nn.Linear(self.num_cam_params, self.cam_feats)
        self.cam_params_preprocess2 = nn.Linear(self.cam_feats, self.cam_feats)
        self.uv_preprocess1 = nn.Linear(2, self.cam_feats)
        self.uv_preprocess2 = nn.Linear(self.cam_feats, self.cam_feats)
        first_layer = nn.Linear(self.cam_feats*2, hidden_size)
        last_layer = nn.Linear(hidden_size, 1)
        middle_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)]
        self.linear_layers = nn.ModuleList([first_layer] + middle_layers + [last_layer])
        
    
    def forward(self, query_points, cam_params):
        # preprocess and append the camera parameters
        cam_params = self.lr(self.cam_params_preprocess1(cam_params))
        cam_params = self.lr(self.cam_params_preprocess2(cam_params))
        uv = self.lr(self.uv_preprocess1(query_points))
        uv = self.lr(self.uv_preprocess2(uv))
        output = torch.hstack([uv, cam_params])
        # pass through each layer
        for i in range(self.num_layers):
            output = self.linear_layers[i](output)
            # LeakyReLU activation
            if i < self.num_layers - 1:
                output = self.lr(output)
            # Sigmoid for last layer
            else:
                output = self.sig(output)
        
        return output

class UnstructuredHybrid(nn.Module):
    '''
    Outputs 2d occ as function of camera parameters and u,v
    Also outputs 3d occ with the additional parameter d (ray depth)
    '''

    def __init__(self, hidden_size=40, num_layers=4, num_cam_params=12):
        super().__init__()
        # store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_cam_params = num_cam_params

        # activation function
        # self.lr = nn.LeakyReLU(0.1)
        self.lr = torch.sin
        self.sig = nn.Sigmoid()

        # other hyperparameters
        self.cam_feats = 20
        self.output_feature_size = 40
        self.depth_feats = 20

        # model parameters
        assert(num_layers >= 2)
        # camera parameters embedding
        self.cam_params_preprocess1 = nn.Linear(self.num_cam_params, self.cam_feats)
        self.cam_params_preprocess2 = nn.Linear(self.cam_feats, self.cam_feats)
        # uv embeddings
        self.uv_preprocess1 = nn.Linear(2, self.cam_feats)
        self.uv_preprocess2 = nn.Linear(self.cam_feats, self.cam_feats)
        # main network
        first_layer = nn.Linear(self.cam_feats*2, hidden_size)
        last_layer = nn.Linear(hidden_size, self.output_feature_size)
        middle_layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-2)]
        self.linear_layers = nn.ModuleList([first_layer] + middle_layers + [last_layer])
        # xyz embedding
        self.xyz_layer1 = nn.Linear(3, self.depth_feats)
        self.xyz_layer2 = nn.Linear(self.depth_feats, self.depth_feats)
        # 2d head
        feat_size = self.output_feature_size + self.depth_feats
        self.head_3d = nn.ModuleList([nn.Linear(feat_size, feat_size)]*(self.num_layers//2 - 1) + [nn.Linear(feat_size,1)])
        self.head_2d = nn.ModuleList([nn.Linear(self.output_feature_size, self.output_feature_size)]* (self.num_layers//2 - 1)\
                                     + [nn.Linear(self.output_feature_size,1)])


    def forward(self, query_points, cam_params, xyz=None):
        # preprocess and append the camera parameters
        cam_params = self.lr(self.cam_params_preprocess1(cam_params))
        cam_params = self.lr(self.cam_params_preprocess2(cam_params))
        uv = self.lr(self.uv_preprocess1(query_points))
        uv = self.lr(self.uv_preprocess2(uv))
        feats = torch.hstack([uv, cam_params])
        # pass through each layer
        for i in range(self.num_layers):
            feats = self.lr(self.linear_layers[i](feats))

        # 2D - use 2d head
        if xyz is None:
            for i in range(self.num_layers//2):
                if i < (self.num_layers//2) - 1:
                    feats = self.lr(self.head_2d[i](feats))
                else:
                    occ = self.sig(self.head_2d[i](feats))
        # 3D - use 3d head
        else:
            # preprocess xyz point
            xyz_feat = self.lr(self.xyz_layer1(xyz))
            xyz_feat = self.lr(self.xyz_layer2(xyz_feat))

            input_3d = torch.hstack([feats, xyz_feat])

            # run 3d head
            for i in range(self.num_layers//2):
                if i < (self.num_layers//2) - 1:
                    input_3d = self.lr(self.head_3d[i](input_3d))
                else:
                    occ = self.sig(self.head_3d[i](input_3d))

        return occ