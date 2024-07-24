# Copyright (c) 2020-2023 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the Open-MiChroM project, released under the MIT License.

R"""
The :class:`~.cndbTools` class perform analysis from **cndb** or **ndb** - (Nucleome Data Bank) file format for storing an ensemble of chromosomal 3D structures.
Details about the NDB/CNDB file format can be found at the `Nucleome Data Bank <https://ndb.rice.edu/ndb-format>`__.
"""


import numpy as np
import h5py
import os

class cndbToolsMini:
    def __init__(self):
        
        self.Type_conversion = {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
        self.Type_conversionInv = {y:x for x,y in self.Type_conversion.items()}
        
    def load(self, filename):
        R"""
        Receives the path to **cndb** or **ndb** file to perform analysis.
        
        Args:
            filename (file, required):
                Path to cndb or ndb file. If an ndb file is given, it is converted to a cndb file and saved in the same directory.
        """
        f_name, file_extension = os.path.splitext(filename)
        
        if file_extension != ".cndb":
            print(f"Please convert the trajectory {filename} into .cndb format") 
            raise TypeError

        self.cndb = h5py.File(filename, 'r')
        
        self.ChromSeq = list(self.cndb['types'])
        self.uniqueChromSeq = set(self.ChromSeq)
        
        self.dictChromSeq = {}
        
        for tt in self.uniqueChromSeq:
            self.dictChromSeq[tt] = ([i for i, e in enumerate(self.ChromSeq) if e == tt])
        
        self.Nbeads = len(self.ChromSeq)
        self.Nframes = len(self.cndb.keys()) -1
        
        return(self)
    
    def xyz(self, frames=[1,None,1], beadSelection=None, XYZ=[0,1,2]):
        R"""
        Get the selected beads' 3D position from a **cndb** or **ndb** for multiple frames.
        
        Args:
            frames (list, required):
                Define the range of frames that the position of the bead will get extracted. The range list is defined by :code:`frames=[initial, final, step]`. (Default value: :code: `[1,None,1]`, all frames)
            beadSelection (list of ints, required):
                List of beads to extract the 3D position for each frame. The list is defined by :code: `beadSelection=[0,1,2,...,N-1]`. (Default value: :code: `None`, all beads) 
            XYZ (list, required):
                List of the axis in the Cartesian coordinate system that the position of the bead will get extracted for each frame. The list is defined by :code: `XYZ=[0,1,2]`. where 0, 1 and 2 are the axis X, Y and Z, respectively. (Default value: :code: `XYZ=[0,1,2]`) 
    
        Returns:
            (:math:`N_{frames}`, :math:`N_{beads}`, 3) :class:`numpy.ndarray`: Returns an array of the 3D position of the selected beads for different frames.
        """
        frame_list = []
        
        if beadSelection == None:
            selection = np.arange(self.Nbeads)
        else:
            selection = np.array(beadSelection)
            
        if frames[1] == None:
            frames[1] = self.Nframes
        
        for i in range(frames[0],frames[1],frames[2]):
            frame_list.append(np.take(np.take(np.array(self.cndb[str(i)]), selection, axis=0), XYZ, axis=1))
        return(np.array(frame_list))
    
    