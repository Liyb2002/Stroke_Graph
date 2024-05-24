import proc_CAD.CAD_to_stroke_cloud
import gnn_graph
import proc_CAD.helper
import os
import SBGCN.run_SBGCN

class single_data:
    def __init__(self, stroke_cloud_graph=None, brep_embedding=None, operations=None):
        self.stroke_cloud= proc_CAD.CAD_to_stroke_cloud.run()
        self.stroke_cloud_graph = gnn_graph.build_graph(self.stroke_cloud)

        file_path = os.path.join(os.getcwd(), 'proc_CAD', 'canvas', 'program.json')

        self.program = proc_CAD.helper.program_to_string(file_path)

        self.SBGCN_encoder = SBGCN.run_SBGCN.load_pretrained_SBGCN_model()




single_data_example = single_data()

