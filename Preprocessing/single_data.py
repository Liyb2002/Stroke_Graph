import proc_CAD.CAD_to_stroke_cloud
import gnn_graph

class single_data:
    def __init__(self, stroke_cloud_graph=None, brep_embedding=None, operations=None):
        self.stroke_cloud= proc_CAD.CAD_to_stroke_cloud.run()
        gnn_graph.build_graph(self.stroke_cloud)




single_data_example = single_data()

